# Mathematical Formalisation — PMSP-SDSC

## Summary

Full OR-style mathematical programming formulation for the **Parallel Machine Scheduling Problem with Sequence-Dependent Setup Costs** (PMSP-SDSC), tailored to the textile fabric dyeing domain. Ready for direct inclusion in a LaTeX thesis via `\include{}`.

---

## 1. Problem Definition

We are given a set of **jobs** (dye batches) that must be assigned to **identical parallel machines** (dyeing vats). Each job has a processing time, a due date, a weight (priority), and a colour defined in a continuous 8-dimensional colour space (Lab + RGB + Cyan + Magenta). The **setup cost** between two consecutive jobs on the same machine depends on the colour distance in this space and on the chemistry class of the dyes used. The objective is a weighted sum of **normalised total weighted tardiness** and **normalised total setup cost**.

---

## 2. LaTeX Source (`formalisation.tex`)

```latex
%=============================================================================
%  PMSP-SDSC FORMALISATION
%  Include: \include{notes/formalisation}
%=============================================================================

\section{Mathematical Programming Formulation}
\label{sec:formalisation}

\subsection{Sets and Indices}

\begin{align}
    \mathcal{J}     &= \{1, \dots, n\}                 && \text{set of jobs} \\
    \mathcal{M}     &= \{1, \dots, m\}                 && \text{set of parallel machines} \\
    \mathcal{K}     &= \{A, D, R, V\}                  && \text{chemistry classes (Acid, Direct, Reactive, Vat)} \\
    \mathcal{C}     &= \{L, a, b, R, G, B, C, M\}      && \text{colour dimensions (Lab + RGB + CM)} \\
    \mathcal{S}     &= \{0, 1, \dots, n\}              && \text{job + dummy job 0 (machine start)}
\end{align}

Each job $j \in \mathcal{J}$ is fully described by the tuple
$\langle p_j, d_j, w_j, \mathbf{o}_j, \kappa_j, \gamma_j \rangle$ where:

\begin{itemize}
    \item $p_j \in \mathbb{R}^+$ — processing time (minutes)
    \item $d_j \in \mathbb{R}^+$ — due date (minutes from time zero)
    \item $w_j \in \mathbb{R}^+$ — weight (priority), default $1$
    \item $\mathbf{o}_j \in [-1,1]^{|\mathcal{C}|}$ — normalised colour observation vector (8-D)
    \item $\kappa_j \in \mathcal{K}$ — chemistry class label
    \item $\gamma_j \in \mathcal{K}$ — vat dye level indicator
\end{itemize}

\subsection{Parameters}

The composite setup cost between job $i$ followed by job $j$ on the same machine is:

\begin{equation}
    c_{ij} = \underbrace{\max(0,\; \mathbf{o}_j^L - \mathbf{o}_i^L)}_{\text{darkness penalty}}
           + \underbrace{\frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} |\mathbf{o}_i^c - \mathbf{o}_j^c|}_{\text{colour distance}}
           + \underbrace{\delta(\kappa_i, \kappa_j)}_{\text{chemistry mismatch}}
           + \underbrace{\epsilon_{ij}}_{\text{noise}}
    \label{eq:setup_cost}
\end{equation}

where:

\begin{equation}
    \delta(\kappa_i, \kappa_j) =
    \begin{cases}
        0 & \text{if } \kappa_i = \kappa_j \\
        0.5 & \text{if compatible classes} \\
        1.0 & \text{if incompatible classes}
    \end{cases}
    \label{eq:chemistry_penalty}
\end{equation}

The noise term $\epsilon_{ij}$ is sampled from a Beta distribution $\mathrm{Beta}(1, 10)$ to simulate machine-specific and operator-specific variability in observed setup times, scaled to $[0, 0.3]$.

\paragraph{Realistic Profile Extension}
The \emph{realistic} problem variant replaces the uniform colour distribution of the \emph{baseline} profile with 12 continuous colour families that mirror real textile production. Each family defines a cluster centre in colour space; jobs from the same family have lower setup costs and are more likely to share a chemistry class. This creates a realistic block-diagonal structure in the setup cost matrix $\mathbf{C}$.

\subsection{Decision Variables}

\begin{align}
    x_{ijk} &\in \{0,1\} && \forall i \in \mathcal{S},\; \forall j \in \mathcal{S} \setminus \{i\},\; \forall k \in \mathcal{M} \\
    y_{jk}  &\in \{0,1\} && \forall j \in \mathcal{J},\; \forall k \in \mathcal{M} \\
    C_j     &\in \mathbb{R}^+ && \forall j \in \mathcal{J} \\
    T_j     &\in \mathbb{R}^+ && \forall j \in \mathcal{J}
\end{align}

where:

\begin{itemize}
    \item $x_{ijk} = 1$ if job $j$ immediately follows job $i$ on machine $k$
    \item $y_{jk} = 1$ if job $j$ is assigned to machine $k$
    \item $C_j$ is the completion time of job $j$
    \item $T_j = \max(0, C_j - d_j)$ is the tardiness of job $j$
\end{itemize}

\subsection{Objective Function}

Minimise the composite score:

\begin{equation}
    F = \alpha \cdot \frac{f_1}{\hat{f}_1} \;+\; (1 - \alpha) \cdot \frac{f_2}{\hat{f}_2}
    \label{eq:objective}
\end{equation}

where $\alpha \in [0,1]$ is a user-specified weight (default $0.5$), and:

\begin{align}
    f_1 &= \sum_{j \in \mathcal{J}} w_j T_j                  && \text{total weighted tardiness} \\
    f_2 &= \sum_{k \in \mathcal{M}} \sum_{i \in \mathcal{S}} \sum_{j \in \mathcal{S} \setminus \{i\}} c_{ij} \, x_{ijk}
                                                              && \text{total setup cost} \\
    \hat{f}_1 &= \frac{1}{R} \sum_{r=1}^{R} f_1^{(r)}       && \text{heuristic-sampled scale for } f_1 \\
    \hat{f}_2 &= \frac{1}{R} \sum_{r=1}^{R} f_2^{(r)}       && \text{heuristic-sampled scale for } f_2
\end{align}

The normalisation constants $\hat{f}_1, \hat{f}_2$ are estimated by running $R = 1000$ random feasible schedules (generated by SPT and random permutation heuristics) and averaging the resulting $f_1$ and $f_2$ values. This heuristic sampling approach provides a robust scale for the composite objective without requiring a priori knowledge of the optimal objective value.

\subsection{Constraints}

\paragraph{Assignment constraints}

\begin{align}
    \sum_{k \in \mathcal{M}} y_{jk} &= 1            && \forall j \in \mathcal{J}
    \label{eq:assign_once} \\
    \sum_{j \in \mathcal{J}} y_{jk} &\geq 1         && \forall k \in \mathcal{M}
    \label{eq:no_idle_machine}
\end{align}

Each job must be assigned to exactly one machine; every machine must receive at least one job (no idle machines).

\paragraph{Sequence constraints}

\begin{align}
    \sum_{i \in \mathcal{S} \setminus \{j\}} x_{ijk} &= y_{jk}      && \forall j \in \mathcal{J},\; \forall k \in \mathcal{M}
    \label{eq:one_predecessor} \\
    \sum_{j \in \mathcal{S} \setminus \{i\}} x_{ijk} &= y_{ik}      && \forall i \in \mathcal{J},\; \forall k \in \mathcal{M}
    \label{eq:one_successor}
\end{align}

Each job has exactly one predecessor and one successor on its assigned machine (or is first/last in the sequence).

\paragraph{Flow conservation (subtour elimination)}

\begin{align}
    \sum_{k \in \mathcal{M}} \sum_{i \in \mathcal{S} \setminus \{j\}} x_{ijk} &= 1     && \forall j \in \mathcal{J}
    \label{eq:flow_cons} \\
    \sum_{j \in \mathcal{J}} x_{0jk} &= 1                                                && \forall k \in \mathcal{M}
    \label{eq:start_from_dummy}
\end{align}

These constraints ensure a Hamiltonian path per machine, starting from dummy job 0.

\paragraph{Completion time and tardiness}

\begin{align}
    C_j &\geq p_j + \sum_{i \in \mathcal{S} \setminus \{j\}} C_i \, x_{ijk}
          + \sum_{i \in \mathcal{S} \setminus \{j\}} c_{ij} \, x_{ijk}
          - M(1 - y_{jk})            && \forall j \in \mathcal{J},\; \forall k \in \mathcal{M}
    \label{eq:completion} \\
    T_j &\geq C_j - d_j              && \forall j \in \mathcal{J}
    \label{eq:tardiness} \\
    T_j &\geq 0                      && \forall j \in \mathcal{J}
    \label{eq:tardiness_nonneg}
\end{align}

where $M$ is a sufficiently large constant (big-M formulation). For the first job on a machine, $C_i = 0$ for the dummy predecessor.

\subsection{Domain-Specific Dyeing Interpretation}

\begin{table}[htbp]
\centering
\caption{Mapping from formal model to textile dyeing domain.}
\label{tab:domain_mapping}
\begin{tabular}{ll}
\toprule
\textbf{Model Symbol} & \textbf{Dyeing Domain Interpretation} \\
\midrule
Job $j$ & Dye batch (roll of grey fabric to be coloured) \\
Machine $k$ & Dyeing vat (industrial vessel with liquor circulation) \\
$p_j$ & Dye cycle time (fill, heat, dwell, drain phases) \\
$d_j$ & Customer deadline or next-process due time \\
$w_j$ & Order priority / customer importance \\
$\mathbf{o}_j$ & Colour specification in Lab+RGB+CM space \\
$\kappa_j$ & Dye chemistry class (direct, reactive, vat, acid) \\
Darkness term $\max(0, \Delta L)$ & Extra cleaning cost when going from light to dark \\
Setup cost $c_{ij}$ & Cleaning + reconfiguration time between batches \\
$T_j$ & Late delivery penalty \\
\bottomrule
\end{tabular}
\end{table}

This formulation enables the comparison of four solution approaches evaluated in this work:
\begin{enumerate}
    \item \textbf{Classical heuristics:} SPT, NN-Greedy
    \item \textbf{Genetic Algorithm (GA):} DEAP-based, order crossover, multi-type mutation
    \item \textbf{GA + PPO:} PPO agent selects mutation operator during GA evolution
\end{enumerate}
```

## 3. Key Design Decisions

1. **Heuristic-sampled normalisation** — uses 1k random/SPT schedules to estimate `f1_scale` and `f2_scale`, avoiding dependency on an optimal solver for normalisation
2. **8-D colour observation** (Lab + RGB + CM) — sufficient for per-job colour discrimination in the scheduling context
3. **Physics-informed setup cost** — the lightness-channel term `max(0, ΔL)` captures the asymmetry in cleaning cost (dark→light vs light→dark)
4. **Domain-variant setup cost** — `realistic` profile clusters jobs into colour families, changing the cost matrix structure to resemble real-world batch patterns

## 4. Suggested Usage

```latex
% In your thesis main .tex file:
\include{notes/formalisation}
```
