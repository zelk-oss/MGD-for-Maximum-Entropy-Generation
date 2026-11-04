"""Generate talk_formulas.tex: each formula is typeset AND printed as copyable LaTeX from the same string."""
from pathlib import Path

OUT = Path(__file__).parent / 'talk_formulas.tex'

# (kind, payload): 'sec' title | 'text' latex paragraph | 'eq' formula string (math-mode content, or a full
# environment if it starts with \begin) | 'note' small grey paragraph (code details, not for the slide)
DOC = [
    ('sec', 'Notation (common to all figures)'),
    ('text', r'$N$ signals $x_b$, $b = 1,\dots,N$, each of $T$ samples. Lags $\tau$ are integers (samples). '
             r'Lagrangian figures show the lag as $\tau/\tau_\eta$ ($\tau_\eta$: Kolmogorov time in samples); '
             r'Eulerian (jets) figures show $\tau$ in samples.'),
    ('eq', r'\delta_\tau x(t) = x(t+\tau) - x(t)'),
    ('text', r'$\langle \cdot \rangle$: average over all signals and over all times $t$ at which the quantity exists. '
             r'$\sigma$: standard deviation,'),
    ('eq', r'\sigma(y)^2 = \langle y^2 \rangle - \langle y \rangle^2'),

    ('sec', r'Structure functions $S_p(\tau)$'),
    ('text', r'Figures: \texttt{li\_structure\_functions} ($p = 2,4,6,8$, Lagrangian); '
             r'\texttt{fig6\_*\_1}, right panel ($p = 4,6,8$, both). Curves shifted down by $10^{-1}$ per order.'),
    ('eq', r'S_p(\tau) = \big\langle (\delta_\tau x)^p \big\rangle'),
    ('note', r'Fig.~6 takes the absolute value of the mean, $|\langle(\delta_\tau x)^p\rangle|$: identical for '
             r'even $p$. Lags: $\tau = 1,\dots,T/2-1$ (Fig.~6), $\tau = 1,\dots,\min(\texttt{max\_lag}, T-2)$ (Li).'),

    ('sec', r'Flatness $F_p(\tau)$'),
    ('text', r'Figure: \texttt{li\_flatness} ($p = 4,6,8$, Lagrangian).'),
    ('eq', r'F_p(\tau) = \frac{S_p(\tau)}{S_2(\tau)^{p/2}}'),
    ('text', r'Gaussian values, for reference (not drawn in the figure):'),
    ('eq', r'F_4 = 3, \qquad F_6 = 15, \qquad F_8 = 105'),

    ('sec', 'Flatness slope'),
    ('text', r'Figure: \texttt{fig6\_*\_2}, left panel (both). Dotted line at $0$.'),
    ('eq', r'\frac{d \log F_4}{d \log \tau}'),
    ('note', r'Evaluated on 30 log-spaced lags in $[1, T/2)$, centred second-order finite differences in '
             r'$\log\tau$ (\texttt{np.gradient}).'),

    ('sec', r'Local scaling exponent $\zeta_p(\tau)$'),
    ('text', r'Figure: \texttt{li\_local\_exponents} ($p = 4,6,8$, Lagrangian). Dotted line at $p/2$ '
             r'(no intermittency: $S_p \propto S_2^{\,p/2}$).'),
    ('eq', r'\zeta_p(\tau) = \frac{d \log S_p}{d \log S_2}'),
    ('note', r'Computed as the ratio $\frac{d \log S_p / d \log \tau}{d \log S_2 / d \log \tau}$ (same thing, chain rule), '
             r'finite differences on the unit-spaced lag grid $\tau = 1, 2, \dots$, as Li et al.\ 2024.'),

    ('sec', 'Increment PDFs'),
    ('eq', r'\mathrm{PDF}\!\left( \frac{\delta_\tau x}{\sigma(\delta_\tau x)} \right)'),
    ('text', r'\texttt{li\_increment\_pdfs} (Lagrangian): normalised density, $\tau/\tau_\eta = 1, 2, 5$, '
             r'shifted down by $10^{-3}$ per lag.\\ '
             r'\texttt{fig6\_*\_2}, middle panel (both): PDF divided by its maximum (peak $=1$), '
             r'$\tau = 1, 4, 16, 64$ samples, shifted down by $10^{-1}$ per lag.'),
    ('note', r'Fig.~6: histogram on $[-6, 6]$, 201 bin edges, Gaussian smoothing of 1.5 bins. '
             r'Li: no smoothing, bins over $\pm\max|\delta_\tau x|/\sigma$.'),

    ('sec', 'Marginal PDF'),
    ('text', r'Figure: \texttt{fig6\_*\_1}, left panel (both).'),
    ('eq', r'\mathrm{PDF}\!\left( \frac{x}{\sigma(x)} \right)'),
    ('note', r'The data are standardised ($\langle x\rangle = 0$, $\sigma(x) = 1$), so the code histograms $x$ '
             r'directly; the model samples are put on the same bins without re-standardising. '
             r'Gaussian smoothing of 1.5 bins.'),

    ('sec', r'Power spectrum $P(\omega)$'),
    ('text', r'Figure: \texttt{fig6\_*\_1}, middle panel (both); $\omega = k/T$ ($k = 1,\dots,T/2$) in cycles per sample, '
             r'shown for $0 < \omega \le 0.4$.'),
    ('eq', r'P(\omega) = \frac{1}{N} \sum_{b=1}^{N} \Big| \sum_{t=0}^{T-1} x_b(t)\, e^{-2\pi i \omega t} \Big|^2'),
    ('note', r'No $1/T$ normalisation (\texttt{|rfft|}$^2$ averaged over signals).'),

    ('sec', 'Skewness'),
    ('text', r'Figure: \texttt{fig6\_*\_2}, right panel (both). Dotted line at $0$.'),
    ('eq', r'\mathrm{skewness}(y) = \frac{\big\langle \big(y - \langle y \rangle\big)^3 \big\rangle}{\sigma(y)^3},'
           r' \qquad y = x(t)\,\delta_\tau x(t)'),
    ('note', r'\texttt{skew\_mode=\textquotesingle power\textquotesingle} (the talk setting). '
             r'45 log-spaced lags in $[1, T/2)$.'),

    ('sec', r'Cross-scale correlations $C_{p,q}(\tau, \tau^\star)$'),
    ('text', r'Figure: \texttt{fig6\_*\_3} (both), $\tau^\star = 20$ samples (dotted vertical line; '
             r'shown as $\tau^\star/\tau_\eta$ in the Lagrangian figure). Horizontal line at $1$: '
             r'value when the increment amplitudes at the two scales are uncorrelated.'),
    ('eq', r'C_{p,q}(\tau, \tau^\star) = '
           r'\frac{\big\langle |\delta_{\tau^\star} x|^p \, |\delta_\tau x|^q \big\rangle}'
           r'{\big\langle |\delta_{\tau^\star} x|^p \big\rangle \, \big\langle |\delta_\tau x|^q \big\rangle}'),
    ('text', r'The three panels are $C_4 \equiv C_{2,2}$, \; $C_6 \equiv C_{3,3}$ and $C_{4,2}$. '
             r'For even $p, q$ the absolute values are not needed, e.g.'),
    ('eq', r'C_4(\tau, \tau^\star) = \frac{\big\langle (\delta_{\tau^\star} x)^2 \, (\delta_\tau x)^2 \big\rangle}'
           r'{S_2(\tau^\star)\, S_2(\tau)}'),
    ('note', r'Both increments start at the same time $t$; the numerator averages over '
             r'$t \le T-1-\max(\tau, \tau^\star)$, each denominator factor over its own full range. '
             r'$10^{-8}$ added to the denominator. Lags $\tau = 1,\dots,T/2-1$.'),

    ('sec', 'Rare signals: acceleration criterion'),
    ('text', r'Acceleration = one-step increment (as Li et al.):'),
    ('eq', r'a(t) = \delta_1 x(t) = x(t+1) - x(t)'),
    ('text', r'Each signal is summarised by its peak acceleration $\max_t |a(t)|$. '
             r'\textbf{Criterion of the talk}: a signal is rare if'),
    ('eq', r'\max_t |a(t)| \;>\; \mathrm{percentile}_{95}\big( \max_t |a| \big)'),
    ('text', r'where the percentile is taken over the data signals: rare = the $5\,\%$ of the data signals with the largest peak acceleration.'),
    ('text', r'\textbf{Alternative} (\texttt{turb\_rare\_typical\_sigma.ipynb}, threshold in units of the standard deviation of $a$): rare if'),
    ('eq', r'\max_t |a(t)| \;\ge\; 20\,\sigma(a)'),
    ('text', r'where $\sigma(a)$ is computed over every time of every data signal. One single number, so the threshold is '
             r'the same level for the data and for the generated signals.'),
    ('note', r'\texttt{rare\_events\_data\_vs\_generated} applies the same rule at the lags $\tau/\tau_\eta = 20, 5, 1$, with $a$ replaced '
             r'by $\delta_\tau x$ and the percentile taken at that lag on the data; the generated signals use the same absolute '
             r'threshold. \texttt{rare\_events\_by\_logp} instead picks signals among the $1\,\%$ with the highest $-\log p$.'),

    ('sec', r'$-\log p$: what the mean and the std are taken over'),
    ('text', r'The model is $p(x) \propto \exp\!\big(+\Theta \cdot \phi(x)\big)$ ($\Theta = \Theta_{\rm reg}$ at the last time step), so'),
    ('eq', r'-\log p(x) = -\,\Theta \cdot \phi(x) + \mathrm{const}'),
    ('text', r'The constant is unknown, so the level of $-\log p$ means nothing: only differences between signals do. '
             r'Large $-\log p$ = unlikely signal (rare signals sit on the right of the histograms).'),
    ('text', r'\textbf{Mean and std are over signals, for one fixed $\Theta$}: $x_b$, $b = 1,\dots,N$, are the $N = 10000$ '
             r'data signals (the training signals):'),
    ('eq', r'\mathrm{mean} = \frac{1}{N} \sum_{b=1}^{N} \big[-\log p(x_b)\big], \qquad '
           r'\mathrm{std} = \Bigg( \frac{1}{N-1} \sum_{b=1}^{N} \Big[-\log p(x_b) - \mathrm{mean}\Big]^2 \Bigg)^{1/2}'),
    ('text', r'\begin{itemize}'
             '\n'
             r'\item \textbf{Table} (per category $c$ of potentials): the same mean and std, applied to the contribution of '
             r'the category $-\sum_{F \in c} \Theta_F \cdot \phi_F(x_b)$ instead of $-\log p(x_b)$. Computed for each seed '
             r'($\Theta$ differs from seed to seed, the data signals are the same), then \emph{averaged over the seeds}.'
             '\n'
             r'\item \textbf{Bar plots}: same quantities; error bars = std over seeds. Third panel: std of the category '
             r'divided by the std of $-\log p$.'
             '\n'
             r'\item \textbf{Histograms}: $-\log p(x_b) - \mathrm{median}$, the median taken over the signals shown '
             r'(all data signals, or all generated signals, each with its own median).'
             '\n'
             r'\item \textbf{Gaussian test}: $-\log p(x_b) - \mathrm{mean}$, the mean taken over the samples of its own '
             r'law (training signals for the exact $-\log p$, generated signals for the MGD one). For $M$ dimensions the exact '
             r'$-\log p = \chi^2_M/2 + \mathrm{const}$, hence $\mathrm{std} = \sqrt{M/2}$.'
             '\n'
             r'\end{itemize}'),
    ('note', r'The sign flip leaves every std unchanged and changes the sign of every mean. Category contributions add up to '
             r'$-\log p + \mathrm{const}$, but their stds do not add up (covariances between categories). '
             r'The mean of a category contribution is a level only: a constant added to $\phi_F$ shifts it without changing $p$. '
             r'Code: \texttt{ddof=1} for every std; \texttt{energy\_table}, \texttt{plot\_energy\_bars}, \texttt{plot\_logp\_histogram}, '
             r'\texttt{plot\_logp\_gaussian} in \texttt{codes/talk\_figures.py}.'),

    ('sec', 'Error bands (all figures)'),
    ('text', r'Line: the statistic on all signals. Band, \texttt{band=\textquotesingle batch\textquotesingle} '
             r'(default): min/max of the same statistic over 10 disjoint batches of signals (data and model, '
             r'as Li et al.). \texttt{band=\textquotesingle seeds\textquotesingle}: model line = mean over '
             r'independent runs, band = $\pm 1$ std; the data band stays min/max over batches.'),
]

HEAD = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{titlesec}
\usepackage{parskip}
\definecolor{codebg}{gray}{0.95}
\definecolor{notegray}{gray}{0.40}
\lstset{basicstyle=\ttfamily\small, backgroundcolor=\color{codebg}, frame=single, rulecolor=\color{gray!40},
        breaklines=true, breakatwhitespace=true, columns=fullflexible, keepspaces=true, upquote=true, aboveskip=2pt, belowskip=8pt}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{14pt}{4pt}
\newcommand{\codenote}[1]{{\small\color{notegray}\emph{Code:} #1\par}}
\begin{document}
{\LARGE\bfseries Statistics in the turbulence talk figures}\par\smallskip
{\small Formulas recovered from \texttt{codes/talk\_figures.py}, \texttt{codes/paper\_figures\_li.py},\\
\texttt{turbulence/paper\_figs\_ensemble.py}, \texttt{codes/check\_moments.py}.\\
Each formula is followed by its LaTeX code (copy from the grey box; math mode, no custom macros).
Grey \emph{Code:} lines are implementation details, not for the slides.}
"""


def emit():
    out = [HEAD]
    for kind, s in DOC:
        if kind == 'sec':
            out.append(f'\\section{{{s}}}')
        elif kind == 'text':
            out.append(s + '\n')
        elif kind == 'note':
            out.append(f'\\codenote{{{s}}}')
        elif kind == 'eq':
            out.append(f'\\[\n{s}\n\\]')
            out.append('\\begin{lstlisting}\n' + s + '\n\\end{lstlisting}')
    out.append('\\end{document}\n')
    OUT.write_text('\n'.join(out))
    print(f'-> {OUT}')


if __name__ == '__main__':
    emit()
