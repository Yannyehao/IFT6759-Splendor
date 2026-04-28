\documentclass{article} % For LaTeX2e
\usepackage{iclr2025_conference,times}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{makecell}
% Optional math commands from https://github.com/goodfeli/dlbook_notation.
\input{math_commands.tex}

\usepackage{hyperref}
\usepackage{url}


\title{Splendor AI: From Reward Shaping to Guided Lookahead Search}

% Authors must not appear in the submitted version. They should be hidden
% as long as the \iclrfinalcopy macro remains commented out below.
% Non-anonymous submissions will be rejected without review.

\author{XinyanGuo, QianyunShen, YehaoYan
}

% The \author macro works with any number of authors. There are two commands
% used to separate the names and addresses of multiple authors: \And and \AND.
%
% Using \And between authors leaves it to \LaTeX{} to determine where to break
% the lines. Using \AND forces a linebreak at that point. So, if \LaTeX{}
% puts 3 of 4 authors names on the first line, and the last on the second
% line, try using \AND instead of \And before the third author name.

\newcommand{\fix}{\marginpar{FIX}}
\newcommand{\new}{\marginpar{NEW}}



%\iclrfinalcopy % Uncomment for camera-ready version, but NOT for submission.
\begin{document}


\maketitle

\begin{abstract}
Developing AI agents for the board game Splendor is challenging due to large discrete action spaces and highly sparse rewards. This project conducts a comparative study between a Score-Based reinforcement learning baseline and an Event-Based reinforcement learning approach enhanced by shallow lookahead planning. Beyond defining a small set of meaningful in-game events, we introduce event-driven reward shaping and use the same event semantics as an inference-time evaluation signal in PPO+Lookahead. Across 1000-game evaluations, Event-Based PPO achieves a 77.9\% win rate against the Greedy baseline, while PPO+Lookahead reaches 91.9\% with $K=15$ and 93.0\% with $K=30$. These results show that high-quality state-action evaluation via semantic events, rather than search depth alone, is the key to strong Splendor play.
\end{abstract}

\section{Introduction}

Splendor is a popular turn-based engine-building board game where players act as Renaissance merchants attempting to acquire gems and transform gems into development cards to attract nobles. Players choose between three primary actions each turn: collecting gem tokens, purchasing development cards using those tokens, or reserving cards for future use.
The game's strategic depth lies in its "engine-building" mechanic: purchased cards provide permanent gemstone discounts, enabling players to acquire increasingly expensive and high-scoring cards as the game progresses. Victory is achieved by being the first to reach 15 prestige points through card acquisitions and noble visits. \\

For Reinforcement Learning, it poses two major challenges:
 a massive, highly dynamic discrete action space (up to 200 dimensions where most actions are illegal), and extremely sparse, delayed rewards. Consequently, standard RL agents struggle to establish the long-term causality between early-game engine building and late-game victory points.

To address these challenges, this project conducts a comparative study evaluating two distinct RL paradigms: a Score-Based framework and an Event-Based framework. 

\section{Baseline Opponents}
\label{sec:baselines}
We evaluate all agents against two non-learning baseline opponents.

RandomAgent. At each turn, the RandomAgent first samples an action type (take gems, buy card, reserve) uniformly at random from the types that currently have at least one legal action, then samples uniformly from all legal actions of that type. It serves as a lower-bound baseline: any agent that cannot consistently beat RandomAgent has failed to learn meaningful game structure.

GreedyAgent. At each turn, the GreedyAgent enumerates all legal actions, simulates each resulting game state via a one-step lookahead, and selects the action that maximises a hand-crafted value heuristic over the successor state. When multiple actions share the highest score, one is chosen uniformly at random to break ties. Because this heuristic captures immediate resource acquisition and victory-point gain, the GreedyAgent plays competent short-term tactics but cannot plan beyond a single move. It constitutes our primary benchmark: consistently beating the GreedyAgent under identical rules is our definition of success, since it requires the learned agent to develop strategies that extend beyond one-step greedy reasoning.


\section{Score-Based Reinforcement Learning}
\label{sec:score_based}

As the baseline of this project, we first develop a Score-Based reinforcement learning agent for Splendor. The main idea of this method is to use the original game information directly, and let the agent learn from score-related rewards, without adding extra event-level guidance. We use this model as the reference system before introducing the Event-Based framework.

\subsection{Feature Engineering}

In our implementation, the Score-Based agent uses a fixed-size state vector with 135 dimensions. The observation is designed to include the most important public and private information in the current game state. Formally, the input can be written as
$$
S_{\text{score}} = [S_{\text{self}} \, \| \, S_{\text{opp}} \, \| \, S_{\text{board}} \, \| \, S_{\text{progress}}] \in \mathbb{R}^{135}.
$$

More specifically, the 135 dimensions are composed of the following parts:
\begin{itemize}
    \item Active player features ($35$ dims): gems, permanent discounts, reserved-card count, victory points, noble count, affordable-card count, and simplified reserved-card features.
    \item Opponent features ($14$ dims): opponent gems, discounts, victory points, noble count, and reserved-card count.
    \item Board gems ($6$ dims): the available gem tokens on the board.
    \item Board cards ($72$ dims): the 12 visible cards on the board, where each card is represented by 6 simplified features.
    \item Board nobles ($6$ dims): simplified noble information.
    \item Game progress ($2$ dims): turn progress and player-perspective indicator.
\end{itemize}

This representation is still score-based because it does not explicitly encode tactical events such as blocking, scarcity taking, or buying a reserved card. Instead, it only provides the raw game state and lets the policy network learn useful patterns by itself.

\subsection{Reward Design}


The reward function in the main setting is
$$
r_t = \Delta \mathrm{Score}_t + 0.01 + 50 \cdot \mathbf{1}_{\mathrm{win}} - 50 \cdot \mathbf{1}_{\mathrm{lose}},
$$
where $\Delta \mathrm{Score}_t$ is the score increase of the learning agent at step $t$, and $\mathbf{1}_{\mathrm{win}}$ and $\mathbf{1}_{\mathrm{lose}}$ are indicator functions for terminal outcomes.

The intuition is straightforward. In this formula, $\Delta \mathrm{Score}_t$ means how much the agent's victory points increase after one action at step $t$. If the action directly gives new points, then this value is positive; otherwise, it is $0$. In this way, the agent can receive an immediate reward signal when it performs a scoring action. The constant $0.01$ gives a very small dense reward, so that the training signal is not completely sparse, and the terminal bonus is used to tell the agent that the final game outcome is also important. Compared with the Event-Based reward shaping, this design is still simpler, but it is more practical than using only the final score.


\subsection{Learning Algorithm}

At the beginning of the project, we trained a standard PPO agent as the first baseline. However, Splendor has a very large discrete action space, while most actions are illegal at a given state. In practice, this caused severe instability, and the agent frequently selected invalid actions.

To solve this problem, we later switched to MaskablePPO. The action space is defined as
$$
A \in \mathbb{Z}^{200},
$$

and at each state $s_t$, the environment provides a legal-action mask
$$
m_t \in \{0,1\}^{200}.
$$
The policy logits are masked before the softmax:
$$
\tilde{l}_{t,i} =
\begin{cases}
l_{t,i}, & m_{t,i} = 1, \\
-\infty, & m_{t,i} = 0.
\end{cases}
$$
Then the policy becomes
$$
\pi_{\theta}(a \mid s_t) = \mathrm{Softmax}(\tilde{l}_t).
$$

This change makes illegal actions impossible by design, and significantly improves training stability. Therefore, the strongest score-based baseline in our project is based on MaskablePPO rather than plain PPO.



\section{Event-Based Reinforcement Learning}
\label{headings}

After building the Score-Based baseline, we further study an Event-Based reinforcement learning method. The main reason is that, in Splendor, many useful actions do not immediately increase victory points, but they are still important for future scoring. Therefore, besides the basic game-state information, we also introduce gem-gap features and event features, so that the agent can receive more structured signals for strategic preparation.

\subsection{Feature Engineering}

The feature vector is defined as
$$
S = [S_{base} \parallel S_{gap} \parallel S_{event}] \in \mathbb{R}^{204}.
$$

Here, $S_{base} \in \mathbb{R}^{135}$ is the same full game-state representation used by the score-based MaskablePPO agent.

$S_{gap} \in \mathbb{R}^{60}$ represents the resource deficit for the 12 visible cards on the board. For one card, let $C$ be its cost vector and $A$ be the player's current total assets (gems + discounts). The gap vector is
$$
g = \max(0, C - A).
$$
These features tell the model how far the player is from buying each visible card.

$S_{event} \in \mathbb{R}^{9}$ records several important in-game events. These events include basic actions such as taking gems, buying cards, and reserving cards, goal-related events such as score increase and reaching the winning threshold, and simple tactical events such as scarcity taking, block reserve, buy reserved, and engine spike. In this way, the model can use a small event representation to capture useful strategic patterns.

\begin{table}[htbp]
  \centering
  \caption{Event Features in the Event-Based Model}
  \begin{tabular}{p{0.08\linewidth} p{0.24\linewidth} p{0.56\linewidth}}
    \toprule
    \textbf{ID} & \textbf{Event} & \textbf{Description} \\
    \midrule
    0 & Take Gems & Player collected gems from the board \\
    1 & Buy Card & Player purchased a card \\
    2 & Reserve & Player reserved a card \\
    3 & Score Up & Action immediately increased victory points \\
    4 & Lethal & Action reached the winning threshold ($\ge 15$ VP) \\
    5 & Scarcity Take & Player took a scarce gem resource \\
    6 & Block Reserve & Player reserved a card when the opponent was close to winning \\
    7 & Buy Reserved & Player bought a previously reserved card \\
    8 & Engine Spike & Action caused a large score increase ($\ge 3$ VP) \\
    \bottomrule
  \end{tabular}
  \label{tab:event_features}
\end{table}

\subsection{Algorithm Architecture}

The Event-Based agent uses the same MaskablePPO architecture as the Score-Based baseline.

\subsection{Reward Shaping}

Since natural rewards in Splendor are very sparse, we add dense event-based reward shaping. The event reward is computed as a weighted sum of the event indicators:
$$
r^{\text{event}}_t = \sum_{i=0}^{8} w_i e_{t,i}.
$$

Here, $e_{t,i}$ indicates whether event $i$ is triggered at step $t$, and $w_i$ is the corresponding reward weight. In our strongest Event-Based PPO baseline, event shaping is combined with the score-progress reward. Larger positive rewards are assigned to events that are closely related to long-term scoring, such as buying cards, increasing score, reaching the winning threshold, and producing an engine spike. Small positive rewards are given to supportive actions such as taking gems, reservation, scarcity taking, and buying reserved cards.



\begin{table}[htbp]
  \centering
  \caption{Weights for Event-Based Reward Shaping}
  \begin{tabular}{clc}
    \toprule
    \textbf{ID} & \textbf{Event} & \textbf{Weight} \\
    \midrule
    0 & Take Gems & 0.01 \\
    1 & Buy Card & 10.0 \\
    2 & Reserve & 0.05 \\
    3 & Score Up & 5.0 \\
    4 & Reach 15 & 25.0 \\
    5 & Scarcity Take & 0.2 \\
    6 & Block Reserve & 1.0 \\
    7 & Buy Reserved & 2.0 \\
    8 & Engine Spike & 5.0 \\
    \bottomrule
  \end{tabular}
  \label{tab:event_reward_weights}
\end{table}

This design encourages the agent to learn not only direct scoring actions, but also useful preparation behaviors for later turns.

\subsection{Comparison with Score-Based}
\begin{table}[h]
\centering
\caption{Performance comparison between the Score-Based and Event-Based agents }
\label{tab:score_vs_event}
\begin{tabular}{llccc}
\toprule
\textbf{Agent} & \textbf{Opponent} & \textbf{Win Rate} & \textbf{95\% CI} & \textbf{Batch Std} \\
\midrule
Score-Based PPO & Random (wrapper)   & 94.8\% &             {[}93.2\%, 96.0\%{]} & $\pm$2.4\%     \\
Score-Based PPO & GreedyAgent   & 75.8\% & {[}73.0\%, 78.4\%{]} & $\pm$4.8\% \\
\midrule
Event-Based PPO & Random (wrapper)   & 94.3\% &            {[}92.7\%, 95.6\%{]}  & $\pm$2.1\%     \\
Event-Based PPO & GreedyAgent   & 77.9\% & {[}75.2\%, 80.4\%{]} & $\pm$2.2\% \\
\bottomrule
\end{tabular}
\end{table}
Table 3 summarises the performance comparison between the Score-Based and Event-Based MaskablePPO agents evaluated over 1000 games against each opponent.
Against the random wrapper opponent, both models perform comparably (94.8\% vs.\ 94.3\%),
confirming that either approach is sufficient to defeat a non-strategic opponent.
The more informative comparison is against the GreedyAgent: the Event-Based
agent achieves a \textbf{77.9\%} win rate versus \textbf{75.8\%} for the
Score-Based baseline, a gain of $+2.1$ percentage points.
The 95\% Wilson confidence intervals overlap slightly ($[73.0\%,\,78.4\%]$ for Score-Based and $[75.2\%,\,80.4\%]$ for Event-Based), so the win-rate gain should be interpreted as modest rather than decisive.
Beyond raw win rate, the Event-Based agent exhibits substantially greater
training stability: batch-level standard deviation decreases from $\pm4.8\%$
to $\pm2.2\%$, a \textbf{54\% reduction in variance}.




\section{Why AlphaZero Does Not Work}
\label{others}
Motivated by the success of AlphaZero in games such as Chess and Go, we also attempted to apply a self-play MCTS-based framework to Splendor. However, AlphaZero-style training failed to approach the MaskablePPO baselines. The best AlphaZero checkpoint in our robust logs reached only 24.0\% against GreedyAgent, and the shaped, warm-started, longer, and combined variants were all at or below 5.0\% against GreedyAgent.

\subsection{Experimental Variants and Results}

We explored several configurations of the AlphaZero approach, as summarized in Table 4. The AlphaZero evaluations used 100-game quick-evaluation sets against Random and Greedy-style opponents.

\subsection{Root Cause Analysis}

We identify two structural reasons why AlphaZero-style methods are poorly suited to Splendor in this setting:

\begin{itemize}
    \item \textbf{Imperfect information.} Splendor involves hidden deck order and stochastic card draws. Standard MCTS assumes a fully observable, deterministic state transition, so its tree search produces value estimates that are systematically biased and unreliable.
    
    \item \textbf{High branching factor and long horizons.} With up to 200 legal actions and games lasting 30--80 turns, the search tree is extremely wide and deep. 50 MCTS simulations per move are insufficient to cover the relevant part of the tree, leading to noisy action selection.
    
\end{itemize}

These findings confirm that simply adding more compute does not fix the structural mismatch between AlphaZero's assumptions and Splendor's game mechanics. This motivates our pivot toward the lookahead planning approach described in the next section.
\begin{table}[h]
\centering
\small
\begin{tabular}{lccl}
\toprule
\textbf{Configuration} & \textbf{vs Random} & \textbf{vs Greedy} & \textbf{Notes} \\
\midrule
Stage A restart & 59.0\% & 24.0\% & Best AlphaZero checkpoint, still far below PPO \\ \addlinespace
Stage B final & 7.0\% & 1.0\% & Training collapsed against both baselines \\ \addlinespace
Event-shaped V3 & 34.0\% & 4.0\% & Best shaped variant \\ \addlinespace
PPO warm-start V4 & 6.0\% & 2.0\% & Policy distillation did not help \\ \addlinespace
Extended V3-long & 10.0\% & 0.0\% & More self-play did not improve Greedy performance \\ \addlinespace
Stage C best effort & 30.0\% & 5.0\% & Combined effort remained far below PPO baseline \\
\bottomrule
\end{tabular}
\caption{AlphaZero variant results.}
\label{table:alphazero_results}
\end{table}

\section{PPO + Lookahead: The Winning Formula}

Based on the failure of MCTS-based methods and the partial success of event-based reward shaping, we designed a hybrid approach that combines the learned policy of our best Event-Based PPO model with a lightweight 1-step lookahead search. This combination avoids the structural problems of deep MCTS while still providing some planning capacity beyond the immediate policy output.

\subsection{Architecture}

The PPO+Lookahead agent operates as follows at inference time. First, all legal actions from the current state are enumerated. The PPO policy network filters these to the top $K=15$ candidates ranked by policy probability. For each candidate action, a 1-step forward simulation is run to obtain the resulting next state. Each resulting state is then scored using a composite evaluation function:

\begin{equation*}
\text{Score} = 0.3 \times \text{PPO\_prob} + 0.5 \times \text{event\_reward} + 0.2 \times \text{future\_value}
\end{equation*}

Here, $\text{PPO\_prob}$ is the normalized policy probability assigned by the PPO network to the action; $\text{event\_reward}$ is the weighted sum of the 9 semantic event signals triggered by the action; and $\text{future\_value}$ is the PPO critic estimate of the resulting state, normalized with a hyperbolic tangent. The action with the highest composite score is selected.

A key design decision is that the lookahead evaluation reuses the same 9 semantic event detectors that guided reward shaping during PPO training. This alignment makes the inference-time evaluator sensitive to the same tactical concepts as the trained policy, while allowing the final action choice to be re-ranked by a small amount of explicit planning.
\begin{table}[h]
\centering
\small
\begin{tabular}{lcc}
\toprule
\textbf{Experiment / Matchup} & \textbf{Win Rate} & \textbf{95\% CI} \\ 
\midrule
PPO alone vs Greedy            & 78.0\% & [75.3\%, 80.5\%] \\ \addlinespace
PPO+Lookahead vs Greedy        & 91.9\% & [90.0\%, 93.4\%] \\ \addlinespace
Lookahead vs PPO (H2H)         & 82.2\% & [79.7\%, 84.5\%] \\ \addlinespace
Self-play (sanity)             & 53.5\% & [46.6\%, 60.3\%] \\ 
\bottomrule
\end{tabular}
\caption{PPO+Lookahead Key Results ($n=1000$). }
\label{table:key_results}
\end{table}
\\Table 5 illustrates the full PPO+Lookahead pipeline and summarizes performance across four evaluation conditions. 
The table reports win rates across four experimental conditions, all evaluated over $n=1000$ games. PPO alone achieves 78.0\% (95\% CI: [75.3\%, 80.5\%]) against GreedyAgent. Adding the lookahead component raises this to 91.9\% ([90.0\%, 93.4\%]), a statistically significant improvement of +14 percentage points. In the Lookahead vs PPO head-to-head (H2H) condition, the lookahead-equipped agent wins 82.2\% ([79.7\%, 84.5\%]) of games, confirming that the improvement is robust across different opponent types and not specific to the fixed GreedyAgent. Finally, the self-play sanity check yields 53.5\% ([46.6\%, 60.3\%]), which is appropriately close to 50\%, validating that the evaluation framework is symmetric and unbiased.
\section{Ablation Study}
\label{sec:ablation}

To understand the contribution of each component in the lookahead scoring formula, we conducted a systematic leave-one-out ablation. Results are shown in Table 6. The most important finding is that removing the event reward signal causes the largest drop in win rate (15.8 percentage points), confirming that semantic event signals are the core of the lookahead's effectiveness. By contrast, removing the PPO value estimate has virtually no effect (+0.9 pp), suggesting that the raw critic value does not add useful information beyond what the policy probability and event signals already capture.
\begin{table}[h]
\centering
\small
\begin{tabular}{lcc}
\toprule
\textbf{Configuration} & \textbf{Win Rate vs Greedy} & \textbf{Delta vs Full System} \\ 
\midrule
Full system (baseline) & 91.9\% & --- \\ \addlinespace
No event reward & 76.1\% & -15.8 pp (Critical) \\ \addlinespace
No PPO value & 92.8\% & +0.9 pp \\ \addlinespace
No PPO prob & 91.0\% & -0.9 pp \\ \addlinespace
Event only & 91.4\% & -0.5 pp \\ \addlinespace
K=5 (fewer candidates) & 87.7\% & -4.2 pp \\ \addlinespace
K=30 (more candidates) & 93.0\% & +1.1 pp \\ \addlinespace
K=50 (too many candidates) & 87.3\% & -4.6 pp \\
\bottomrule
\end{tabular}
\caption{Component ablation for PPO+Lookahead.}
\label{table:ablation_study}
\end{table}
For the candidate set size $K$, a sweet spot exists between $K=15$ and $K=30$. Smaller $K$ values ($K=5$) restrict the search to too few options, causing the agent to miss high-value alternatives. Larger values ($K=50$) introduce low-quality candidates that dilute the scoring signal. The optimal configuration in our experiments is $K=30$, which achieves a 93.0\% win rate against GreedyAgent.

\subsection{Event Signal Importance}

\begin{table}[h]
\centering
\small
\begin{tabular}{lccl}
\toprule
\textbf{Event Removed} & \textbf{Win Rate vs Greedy} & \textbf{Delta (pp)} & \textbf{Interpretation} \\ 
\midrule
buy\_card       & 86.4\% & -5.5 & Most critical (MVP) \\ \addlinespace
reach\_15       & 89.4\% & -2.5 & Win detection essential \\ \addlinespace
buy\_reserved   & 90.4\% & -1.5 & Useful but secondary \\ \addlinespace
scarcity\_take  & 90.6\% & -1.3 & Tactical value \\ \addlinespace
engine\_spike   & 90.8\% & -1.1 & Mild contribution \\ \addlinespace
block\_reserve  & 91.2\% & -0.7 & Small positive \\ \addlinespace
score\_up       & 92.4\% & +0.5 & Slight noise \\ \addlinespace
take\_gems      & 92.6\% & +0.7 & Noise (removing helps) \\ \addlinespace
reserve\_card   & 93.0\% & +1.1 & Noise (removing helps) \\ 
\bottomrule
\end{tabular}
\caption{Leave-one-out event signal ablation for PPO+Lookahead. Evaluated over 500 games per condition.}
\label{table:event_ablation}
\end{table}

We further performed a leave-one-out ablation over the 9 individual event signals within the lookahead evaluation. As shown in Table 7, \texttt{buy\_card} is the single most critical signal: removing it drops win rate by 5.5 percentage points to 86.4\%. The \texttt{reach\_15} signal (winning threshold detection) is the second most important (-2.5 pp). Interestingly, removing the \texttt{reserve\_card} signal improves win rate by 1.1 pp in this 500-game ablation, suggesting that reservation signals may introduce noise in the 1-step lookahead setting, possibly because reserving a card has indirect, multi-step benefits that are not well captured in a single-step evaluation.

\section{Results and Discussion}

\subsection{Comparative Performance Against Baselines}
The experimental results demonstrate a clear evolutionary trajectory of the agent's performance. As shown in Table~\ref{tab:final_results}, score-based reward shaping already produces a competent MaskablePPO policy, event shaping makes the policy slightly stronger and more stable, and the integration of 1-step lookahead produces the largest improvement.

\begin{table}[h]
\centering
\caption{Evidence-backed performance summary from 1000-game evaluations.}
\label{tab:final_results}
\begin{tabular}{lccc}
\toprule
\textbf{Agent Architecture} & \textbf{vs Random (\%)} & \textbf{vs Greedy (\%)} & \textbf{95\% CI vs Greedy} \\
\midrule
Score-Based PPO & 94.8 & 75.8 & [73.0, 78.4] \\
Event-Based PPO & 94.3 & 77.9 & [75.2, 80.4] \\
PPO + Lookahead ($K=15$) & 97.8 & 91.9 & [90.0, 93.4] \\
\textbf{PPO + Lookahead ($K=30$)} & \textbf{96.9} & \textbf{93.0} & \textbf{[91.2, 94.4]} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Human-AI Playtest Analysis}
To evaluate the agent in a non-synthetic setting, we also inspected the \texttt{web\_game\_history.jsonl} logs from 111 human-facing web games. Among 47 games against the PPO+Lookahead agent, the AI won 17 games and averaged 10.3 final points overall; in the 46 lookahead games with turn counts recorded, the average game length was 29.5 turns. These logs are not a controlled benchmark because the human opponents and game conditions are not standardized, but they provide a useful qualitative sanity check that the agent plays complete, competitive games in the web interface.

\section{Conclusion}
The success of the Event-Based PPO and its hybrid Lookahead variant leads to our core thesis: \textbf{Evaluation Quality $>$ Search Depth}. 

In environments characterized by high stochasticity and hidden information, such as Splendor's deck mechanics, brute-force search algorithms like AlphaZero fail due to shallow tree coverage and "hallucination." In contrast, our approach utilizes:
\begin{enumerate}
    \item \textbf{High-Quality Perception:} Using the 204-dim feature vector, including gem-gap features, to expose resource affordability directly to the policy.
    \item \textbf{Tactical Intuition:} Guiding the search using semantic event rewards that understand \textit{why} a move matters, not just its point value.
\end{enumerate}

As our findings confirm: if a reward function only recognizes the final score, the agent only learns to chase points. But when the reward function incorporates tactical intent, the agent masters the strategy. Future research will explore Information Set MCTS (ISMCTS) to further refine the agent's performance under deck uncertainty.

\vspace{0.5em}
{\small Code repository: \url{https://github.com/Yannyehao/IFT6759-Splendor}}






\end{document}
