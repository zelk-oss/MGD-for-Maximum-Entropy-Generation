#!/usr/bin/env python3
"""PreToolUse guard for Python run through Bash in this project.

Why (2026-10-28): `Bash(python *)` / `Bash(python3 *)` are allowed in .claude/settings.local.json, so
Python runs without a prompt. The command-text deny rules (ssh, scp, rsync, sbatch, git push, the rm
guard, the Edit denies on results) never look INSIDE a script, so a Python script could do any of those.
This hook reads the Python that is about to run and refuses clearly dangerous behaviour; everything
else gets no decision (so it keeps running without a prompt).

What is scanned: the script file named in the command (python3 script.py), the -c argument, or the whole
command when the code comes from stdin / a heredoc / a file that cannot be read (covers a script created
in the same command). NOT scanned: modules the script imports (repo modules such as codes/utils.py
legitimately write results), `python -m module`.

DENY when the code
  1. spawns a process (subprocess, os.system, os.popen, os.exec*, os.spawn*, pty.spawn) AND names ssh, scp,
     sftp, rsync, sbatch, scancel, squeue, srun, `git push`, the push/pull scripts; or imports paramiko /
     asyncssh / fabric / pexpect;
  2. deletes: recursively (shutil.rmtree, os.removedirs, send2trash) unless the code names the Claude
     scratchpad area; single files (os.remove/unlink/rmdir, .unlink(), .rmdir()) when the same line names
     saved_results/data_files/experiments/.git or a quoted absolute/~ path outside the scratchpad and /tmp
     (deleting a temp file through a variable passes);
  3. writes (open(...,'w'/'a'/'x'), torch.save, np.save*, shutil.copy*/move, os.rename/replace,
     .write_text/.write_bytes, .to_csv/.to_pickle, json.dump, pickle.dump) on a line that names
     saved_results, data_files or experiments/.../figures;
  4. names a secret: Claude credential/account/history files, ~/.ssh, id_rsa/ed25519/ecdsa, *.pem, *.key,
     .env, ~/.aws, ~/.netrc, ~/.git-credentials, ~/.gnupg, ~/.config/gh;
  5. obfuscates: __import__(, or exec/eval together with base64/zlib/marshal/codecs/binascii.
Accident guard, not a sandbox: variable indirection (path = ...; open(path, 'w')) or an imported module
gets past it. If the hook itself fails on a command that runs python, it denies (fails closed).
"""
import json
import os
import re
import shlex
import sys

SCRATCH = '/tmp/claude-1000/-home-chiaraz-phd-MGD-for-Maximum-Entropy-Generation'
PY = re.compile(r'^python[0-9.]*$')
OPERATORS = {';', '&&', '||', '|', '&', '(', ')', '<', '>', '>>', '<<', '|&', ';;'}
VALUE_FLAGS = {'-W', '-X', '--check-hash-based-pycs'}
MAX_BYTES = 2_000_000

SPAWN = re.compile(r'\b(subprocess|os\.system|os\.popen|os\.exec\w*|os\.spawn\w*|pty\.spawn|create_subprocess_\w+)\b')
FORBIDDEN_CMD = re.compile(r'\b(ssh|scp|sftp|rsync|sbatch|scancel|squeue|srun)\b|git\s+push|scripts/(push|pull)|pull_results')
SSH_LIBS = re.compile(r'\b(paramiko|asyncssh|fabric|pexpect)\b')
RECURSIVE_DELETE = re.compile(r'\bshutil\.rmtree\b|\bos\.removedirs\b|send2trash')
FILE_DELETE = re.compile(r'\bos\.(remove|unlink|rmdir)\b|\.unlink\(|\.rmdir\(')
QUOTED_ABS = re.compile(r"""['"]((?:/|~)[^'"]*)['"]""")
PROTECTED_DIRS = re.compile(r'saved_results|data_files|experiments|\.git\b')
WRITE = re.compile(r"open\([^)]*['\"][wax]|torch\.save|np\.save|numpy\.save|shutil\.(copy\w*|move)|os\.(rename|replace)"
                   r"|\.write_(text|bytes)|\.to_(csv|pickle)|json\.dump|pickle\.dump")
RESULTS = re.compile(r'saved_results|data_files|experiments/[^\s\'"]*figures')
SECRETS = re.compile(r'\.credentials\.json|\.claude\.json|\.claude/history\.jsonl|\.ssh\b|\bid_(rsa|ed25519|ecdsa)\b'
                     r'|\.pem\b|\.key\b|\.env\b|\.aws\b|\.netrc|\.git-credentials|\.gnupg|\.config/gh')
OBFUSC = re.compile(r'__import__\s*\(')
EXEC_EVAL = re.compile(r'\b(exec|eval)\s*\(')
DECODERS = re.compile(r'\b(base64|zlib|marshal|codecs|binascii)\b')
RAW = ''


def deny(reason):
    print(json.dumps({'hookSpecificOutput': {
        'hookEventName': 'PreToolUse', 'permissionDecision': 'deny',
        'permissionDecisionReason': f'Python guard (.claude/hooks/guard_python.py): {reason}. Tell the user what '
                                    'you wanted to do and why; they can run it themselves.'}}))
    sys.exit(0)


def scan(code, origin):
    """Deny on the first dangerous pattern in `code`."""
    if SPAWN.search(code) and FORBIDDEN_CMD.search(code):
        deny(f'{origin} spawns a process and names ssh/scp/rsync/sbatch/git push')
    if SSH_LIBS.search(code):
        deny(f'{origin} uses an ssh library')
    if RECURSIVE_DELETE.search(code) and SCRATCH not in code:
        deny(f'{origin} deletes recursively without naming the scratchpad area')
    for line in code.splitlines():
        if FILE_DELETE.search(line):
            outside = [p for p in QUOTED_ABS.findall(line) if not p.startswith((SCRATCH, '/tmp/'))]
            if PROTECTED_DIRS.search(line) or outside:
                deny(f'{origin} deletes a protected or out-of-scratchpad path: {line.strip()[:80]!r}')
        if WRITE.search(line) and RESULTS.search(line):
            deny(f'{origin} writes to saved_results/data_files/figures: {line.strip()[:80]!r}')
    m = SECRETS.search(code)
    if m:
        deny(f'{origin} names a secret file ({m.group(0)})')
    if OBFUSC.search(code) or (EXEC_EVAL.search(code) and DECODERS.search(code)):
        deny(f'{origin} uses obfuscated execution (__import__, exec/eval with a decoder)')


WRAPPERS = {'timeout', 'nohup', 'time', 'env', 'sudo', 'command', 'exec', 'nice', 'stdbuf', 'ionice'}
ASSIGN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')


def at_command_position(toks, i):
    """True if toks[i] can start a command: first token, after an operator, or after wrappers / VAR=value /
    their flags and numbers (timeout 300 python3 ..., env -i X=1 python3 ...). Prose inside a heredoc body
    (the word 'python' in a note) is not a command."""
    j = i - 1
    while j >= 0:
        t = toks[j]
        if t in OPERATORS:
            return True
        if os.path.basename(t) in WRAPPERS:
            return True
        if ASSIGN.match(t) or t.startswith('-') or re.match(r'^[0-9.]+[smhd]?$', t):
            j -= 1
            continue
        return False
    return True


def split_newlines(cmd):
    """A newline outside quotes ends a command: make it a `;` (a backslash-newline joins lines). Newlines inside
    quotes (multi-line python -c "...") are kept so line-based checks still see separate lines."""
    out, quote, i = [], None, 0
    while i < len(cmd):
        c = cmd[i]
        if c == '\\' and i + 1 < len(cmd) and quote != "'":
            if cmd[i + 1] == '\n' and quote is None:
                i += 2
                continue
            out.append(cmd[i:i + 2])
            i += 2
            continue
        if quote is None and c in '\'"':
            quote = c
        elif quote == c:
            quote = None
        out.append(' ; ' if (c == '\n' and quote is None) else c)
        i += 1
    return ''.join(out)


def python_invocations(toks):
    """(index of the python token, its argument list) for every python in the command."""
    out = []
    for i, t in enumerate(toks):
        if PY.match(os.path.basename(t)) and at_command_position(toks, i):
            j = i + 1
            while j < len(toks) and toks[j] not in OPERATORS:
                j += 1
            out.append(toks[i + 1:j])
    return out


def main():
    global RAW
    RAW = sys.stdin.read()
    data = json.loads(RAW)
    if data.get('tool_name') != 'Bash':
        return
    cmd = (data.get('tool_input') or {}).get('command') or ''
    if 'python' not in cmd:
        return
    cwd = data.get('cwd') or os.getcwd()
    try:
        lex = shlex.shlex(split_newlines(cmd), posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError:
        scan(cmd, 'the command (unparsable)')
        return
    for args in python_invocations(toks):
        script, j, whole = None, 0, False
        while j < len(args):
            a = args[j]
            if a == '-c' and j + 1 < len(args):
                scan(args[j + 1], 'the -c code')
                break
            if a == '-m':
                break
            if a == '-':
                whole = True
                break
            if a in VALUE_FLAGS:
                j += 2
                continue
            if a.startswith('-'):
                j += 1
                continue
            script = a
            break
        else:
            whole = True                                             # no script: code from stdin / heredoc
        if '<<' in cmd:
            whole = True                                             # a heredoc may (re)write the script
        if script is not None:
            p = script if os.path.isabs(script) else os.path.join(cwd, script)
            try:
                with open(p, errors='replace') as f:
                    scan(f.read(MAX_BYTES), f'script {os.path.basename(p)}')
            except OSError:
                whole = True                                         # created in this command, or unreadable
        if whole:
            scan(cmd, 'the command text')


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:                                           # fail closed for python commands
        if re.search(r'\bpython', RAW):
            deny(f'guard error ({type(e).__name__}); refusing to be safe')
