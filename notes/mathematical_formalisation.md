# Mathematical Formalisation — PMSP-SDSC

## Summary

Full OR-style mathematical programming formulation for the **Parallel Machine Scheduling Problem with Sequence-Dependent Setup Costs** (PMSP-SDSC), tailored to the textile fabric dyeing domain. Ready for direct inclusion in a LaTeX thesis via `\include{}`.

---

## 1. Problem Definition

We are given a set of **jobs** (dye batches) that must be assigned to **identical parallel machines** (dyeing vats). Each job has a processing time, a due date, a weight, and a categorical colour level (1-7, white to black). The **setup cost** between two consecutive jobs on the same machine depends on the darkness difference: transitioning from a dark colour to a light colour requires expensive deep cleaning, while light-to-dark transitions are cheap. The objective is a weighted sum of **normalised total weighted tardiness** and **normalised total setup cost**.

### Weekly Capacity Model

Processing times are calibrated so the total workload fills one week of continuous operation:

- Each machine operates $H = 168$ hours per week (24/7).
- Total weekly capacity across $m$ machines is $m \times H$.
- For $n$ jobs, the mean processing time is $\bar{p} = (m \times H) / n$, ensuring $\sum p_j \approx m \times H$.
- Due dates are spread across the week: the $k$-th job in SPT order is due at approximately $(k/n) \times H$ hours, creating proportional deadlines throughout the week.
- Setup time between jobs averages $1/8$ of the mean processing time, reflecting the ratio of vat cleaning time (1-2 hours) to dye cycle time (8-16 hours) in textile operations.

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
    \mathcal{K}     &= \{1, \dots, 7\}                 && \text{colour darkness levels (1=white, 7=black)} \\
    \mathcal{S}     &= \{0, 1, \dots, n\}              && \text{job + dummy job 0 (machine start)}
\end{align}

Each job $j \in \mathcal{J}$ is described by $\langle p_j, d_j, w_j, \ell_j \rangle$ where:
\begin{itemize}
    \item $p_j \in \mathbb{R}^+$ — processing time (hours)
    \item $d_j \in \mathbb{R}^+$ — due date (hours from time zero), proportional to position in SPT order within a 168-hour week
    \item $w_j \in \mathbb{R}^+$ — weight (priority), $w_j = 1$ for all jobs
    \item $\ell_j \in \mathcal{K}$ — categorical colour darkness level
\end{itemize}

\subsection{Parameters}

The setup cost between job $i$ followed by job $j$ on the same machine is asymmetric:

\begin{equation}
    c_{ij} =
    \begin{cases}
        10 \times (\ell_i - \ell_j) + \epsilon_{ij} & \text{if } \ell_i > \ell_j \text{ (dark $\to$ light)} \\
        3 \times (\ell_j - \ell_i) + \epsilon_{ij}  & \text{if } \ell_i < \ell_j \text{ (light $\to$ dark)} \\
        \epsilon_{ij}                                 & \text{if } \ell_i = \ell_j
    \end{cases}
    \label{eq:setup_cost}
\end{equation}

where $\epsilon_{ij} \sim U(0, 2)$ is a small noise term. The dark-to-light case is penalised more heavily because draining pigment residue from a dark batch requires a full vat cleaning cycle. The setup time $s_{ij}$ (in hours) is proportional to the setup cost, scaled so that $\mathbb{E}[s_{ij}] = \bar{p} / 8$, i.e., the average setup time is one-eighth of the average processing time.

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

The normalisation constants $\hat{f}_1, \hat{f}_2$ are estimated by running $R$ heuristic schedules (SPT, NN-Greedy, random) and taking 1.5$\times$ the maximum observed value. This ensures both objectives contribute comparably to the composite without requiring an optimal solver.

\subsection{Constraints}

\paragraph{Assignment constraints}

\begin{align}
    \sum_{k \in \mathcal{M}} y_{jk} &= 1            && \forall j \in \mathcal{J}
    \label{eq:assign_once} \\
    \sum_{j \in \mathcal{J}} y_{jk} &\geq 1         && \forall k \in \mathcal{M}
    \label{eq:no_idle_machine}
\end{align}

Each job assigned to exactly one machine; every machine gets at least one job.

\paragraph{Sequence constraints}

\begin{align}
    \sum_{i \in \mathcal{S} \setminus \{j\}} x_{ijk} &= y_{jk}      && \forall j \in \mathcal{J},\; \forall k \in \mathcal{M}
    \label{eq:one_predecessor} \\
    \sum_{j \in \mathcal{S} \setminus \{i\}} x_{ijk} &= y_{ik}      && \forall i \in \mathcal{J},\; \forall k \in \mathcal{M}
    \label{eq:one_successor}
\end{align}

\paragraph{Flow conservation (subtour elimination)}

\begin{align}
    \sum_{k \in \mathcal{M}} \sum_{i \in \mathcal{S} \setminus \{j\}} x_{ijk} &= 1     && \forall j \in \mathcal{J}
    \label{eq:flow_cons} \\
    \sum_{j \in \mathcal{J}} x_{0jk} &= 1                                                && \forall k \in \mathcal{M}
    \label{eq:start_from_dummy}
\end{align}

\paragraph{Completion time and tardiness}

\begin{align}
    C_j &\geq p_j + \sum_{i \in \mathcal{S} \setminus \{j\}} C_i \, x_{ijk}
          + \sum_{i \in \mathcal{S} \setminus \{j\}} s_{ij} \, x_{ijk}
          - M(1 - y_{jk})            && \forall j \in \mathcal{J},\; \forall k \in \mathcal{M}
    \label{eq:completion} \\
    T_j &\geq C_j - d_j              && \forall j \in \mathcal{J}
    \label{eq:tardiness} \\
    T_j &\geq 0                      && \forall j \in \mathcal{J}
    \label{eq:tardiness_nonneg}
\end{align}

where $M$ is a sufficiently large constant (big-M formulation).

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
$d_j$ & Customer deadline or due time within the weekly schedule \\
$w_j$ & Order priority (all equal in baseline) \\
$\ell_j$ & Colour darkness level (1=white, 7=black) \\
Dark-to-light cost premium & Extra cleaning cost when pigment residue must be fully removed \\
Setup cost $c_{ij}$ & Cleaning cost between batches \\
Setup time $s_{ij}$ & Cleaning duration (avg 1/8 of dye cycle) \\
$T_j$ & Late delivery penalty \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Instance Generation Justification}

All synthetic instances are generated under the following physically grounded assumptions:

\begin{enumerate}
    \item \textbf{Constant production rate:} Dyeing machines operate at a fixed liquor flow rate, so job processing time is proportional to fabric quantity.
    \item \textbf{Weekly capacity:} Total processing time across $n$ jobs fills exactly $m \times 168$ machine-hours, ensuring feasible schedules exist within a one-week horizon.
    \item \textbf{Due date proportionality:} Due dates are distributed across the week in SPT order, creating a range of tight and slack deadlines that reward intelligent sequencing.
    \item \textbf{Asymmetric setup:} Dark-to-light transitions cost 3-4$\times$ more than light-to-dark, reflecting the physics of vat cleaning.
    \item \textbf{Setup/processing ratio:} Setup time averages one-eighth of processing time, matching industrial ratios (1-2 hrs cleaning vs 8-16 hrs dyeing).
\end{enumerate}

This formulation enables the comparison of three solution approaches evaluated in this work:
\begin{enumerate}
    \item \textbf{Classical heuristics:} SPT, NN-Greedy
    \item \textbf{Genetic Algorithm (GA):} DEAP-based, order crossover, multi-type mutation
    \item \textbf{GA + PPO:} PPO agent selects mutation operator during GA evolution
\end{enumerate}
```

## 3. Key Design Decisions

1. **Heuristic-sampled normalisation** — uses SPT/NN/random schedules to estimate `f1_scale` and `f2_scale`, avoiding dependency on an optimal solver
2. **Darkness-asymmetric setup cost** — the only domain physics in the model; dark→light penalty drives scheduling decisions
3. **Weekly capacity grounding** — processing times, due dates, and setup times all derive from a single weekly-hours parameter, giving all parameters a physically interpretable meaning

## 4. Suggested Usage

```latex
% In your thesis main .tex file:
\include{notes/formalisation}
```