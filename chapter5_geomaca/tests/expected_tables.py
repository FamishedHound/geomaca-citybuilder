"""Rows (header included) and captions of the Chapter 5 tables, and the bar heights
of Figure 5.3, as printed in the thesis."""

TABLE_ROWS = {
    'table5.2': [
        '\\textbf{Pillar} & \\textbf{Metric} & \\textbf{Heuristic} & \\textbf{QMIX} & \\textbf{COMA} & \\textbf{GeoMACA} \\\\',
        '\\multirow{3}{*}{\\textbf{Economy}} & GDP (\\$M $\\uparrow$) & 236.7{\\scriptsize$\\pm$0.56}$^{*}$ & 234.5{\\scriptsize$\\pm$0.65}$^{**}$ & 230.1{\\scriptsize$\\pm$1.06}$^{****}$ & \\textbf{241.9{\\scriptsize$\\pm$1.67}} \\\\',
        ' & Tax Revenue (\\$M $\\uparrow$) & 2.2{\\scriptsize$\\pm$0.00}$^{**}$ & 2.2{\\scriptsize$\\pm$0.06}$^{**}$ & 1.9{\\scriptsize$\\pm$0.01}$^{**}$ & \\textbf{7.4{\\scriptsize$\\pm$1.47}} \\\\',
        ' & Developer Profit (\\$M $\\uparrow$) & 474.8{\\scriptsize$\\pm$1.44}$^{*}$ & 474.3{\\scriptsize$\\pm$2.59}$^{*}$ & 477.2{\\scriptsize$\\pm$1.52} & \\textbf{484.5{\\scriptsize$\\pm$3.24}} \\\\',
        '\\multirow{3}{*}{\\textbf{Environ.}} & Green ratio (\\% $\\uparrow$) & 16.9{\\scriptsize$\\pm$0.02}$^{**}$ & 23.2{\\scriptsize$\\pm$1.27} & \\textbf{29.5{\\scriptsize$\\pm$0.48}}$^{****}$ & 20.5{\\scriptsize$\\pm$1.06} \\\\',
        ' & Woodland ratio (\\% $\\uparrow$) & 16.7{\\scriptsize$\\pm$0.03} & 19.4{\\scriptsize$\\pm$0.83} & \\textbf{24.5{\\scriptsize$\\pm$0.46}}$^{***}$ & 18.1{\\scriptsize$\\pm$1.30} \\\\',
        ' & Density (\\% $\\downarrow$) & 46.1{\\scriptsize$\\pm$0.02}$^{**}$ & 46.0{\\scriptsize$\\pm$0.02} & \\textbf{46.0{\\scriptsize$\\pm$0.02}} & 46.0{\\scriptsize$\\pm$0.02} \\\\',
        '\\multirow{3}{*}{\\textbf{Society}} & Population $\\uparrow$ & 1587{\\scriptsize$\\pm$2}$^{*}$ & 1570{\\scriptsize$\\pm$5}$^{***}$ & 1526{\\scriptsize$\\pm$6}$^{****}$ & \\textbf{1603{\\scriptsize$\\pm$5}} \\\\',
        ' & Segregation $\\downarrow$ & 0.093{\\scriptsize$\\pm$0.001} & 0.094{\\scriptsize$\\pm$0.001} & \\textbf{0.091{\\scriptsize$\\pm$0.001}}$^{***}$ & 0.095{\\scriptsize$\\pm$0.001} \\\\',
        ' & Quality of Life $\\uparrow$ & 5.069{\\scriptsize$\\pm$0.001}$^{**}$ & 5.094{\\scriptsize$\\pm$0.005} & 5.116{\\scriptsize$\\pm$0.002} & \\textbf{5.123{\\scriptsize$\\pm$0.013}} \\\\',
        '\\multicolumn{2}{l}{\\textbf{Metric wins}} & 0/9 & 0/9 & 4/9 & \\textbf{5/9} \\\\',
    ],
    'table5.3': [
        '\\textbf{Method} & \\textbf{Top-2} & \\textbf{Avg rank} & \\textbf{Eco $z$} & \\textbf{Soc $z$} & \\textbf{Env $z$} & \\textbf{Overall} \\\\',
        'Heuristic & 3/9 & 3.11 & -0.31 & -0.26 & -1.13 & -0.57 \\\\',
        'QMIX & 3/9 & 2.78 & -0.48 & -0.24 & +0.05 & -0.22 \\\\',
        'COMA & 6/9 & 2.22 & -0.61 & +0.21 & +1.17 & +0.26 \\\\',
        'GeoMACA & 6/9 & 1.89 & +1.40 & +0.29 & -0.10 & +0.53 \\\\',
    ],
    'table5.4': [
        '\\textbf{Configuration} & \\textbf{Population $\\uparrow$} & \\textbf{QoL $\\uparrow$} & \\textbf{Profit (\\$M) $\\uparrow$} & \\textbf{Segregation $\\downarrow$} \\\\',
        'Uniform credit & 1598{\\scriptsize$\\pm$15} & \\textbf{5.145{\\scriptsize$\\pm$0.015}} & \\textbf{488.6{\\scriptsize$\\pm$3.46}} & \\textbf{0.094{\\scriptsize$\\pm$0.001}} \\\\',
        'Temporal only & \\textbf{1605{\\scriptsize$\\pm$3}} & 5.125{\\scriptsize$\\pm$0.007} & 484.9{\\scriptsize$\\pm$2.33} & 0.095{\\scriptsize$\\pm$0.001} \\\\',
        'Temporal + Agent & 1587{\\scriptsize$\\pm$18} & 5.121{\\scriptsize$\\pm$0.012} & 482.1{\\scriptsize$\\pm$3.25} & 0.094{\\scriptsize$\\pm$0.002} \\\\',
        'Full GeoMACA & 1604{\\scriptsize$\\pm$6} & 5.122{\\scriptsize$\\pm$0.011} & 485.5{\\scriptsize$\\pm$3.98} & 0.095{\\scriptsize$\\pm$0.001} \\\\',
    ],
    'table5.5': [
        '\\textbf{Pillar} & \\textbf{Metric} & \\textbf{Heuristic} & \\textbf{No government} & \\textbf{Paired difference} \\\\',
        '\\multirow{3}{*}{\\textbf{Economy}} & GDP (\\$M $\\uparrow$) & 236.7{\\scriptsize$\\pm$0.56} & 235.3{\\scriptsize$\\pm$0.54} & $-1.36${\\scriptsize$\\pm$0.02}$^{****}$ \\\\',
        ' & Tax Revenue (\\$M $\\uparrow$) & 2.2{\\scriptsize$\\pm$0.00} & 0.0{\\scriptsize$\\pm$0.00} & $-2.16${\\scriptsize$\\pm$0.00}$^{****}$ \\\\',
        ' & Developer Profit (\\$M $\\uparrow$) & 474.8{\\scriptsize$\\pm$1.44} & 473.3{\\scriptsize$\\pm$1.38} & $-1.46${\\scriptsize$\\pm$0.06}$^{****}$ \\\\',
        '\\multirow{3}{*}{\\textbf{Environ.}} & Green ratio (\\% $\\uparrow$) & 16.9{\\scriptsize$\\pm$0.02} & 16.7{\\scriptsize$\\pm$0.02} & $-0.16${\\scriptsize$\\pm$0.01}$^{****}$ \\\\',
        ' & Woodland ratio (\\% $\\uparrow$) & 16.7{\\scriptsize$\\pm$0.03} & 16.7{\\scriptsize$\\pm$0.03} & $0$ \\\\',
        ' & Density (\\% $\\downarrow$) & 46.1{\\scriptsize$\\pm$0.02} & 46.0{\\scriptsize$\\pm$0.01} & $0$ \\\\',
        '\\multirow{3}{*}{\\textbf{Society}} & Population $\\uparrow$ & 1587{\\scriptsize$\\pm$2} & 1583{\\scriptsize$\\pm$2} & $-4.4${\\scriptsize$\\pm$0.08}$^{****}$ \\\\',
        ' & Segregation $\\downarrow$ & 0.093{\\scriptsize$\\pm$0.001} & 0.093{\\scriptsize$\\pm$0.001} & $-0.0001${\\scriptsize$\\pm$0.0001} \\\\',
        ' & Quality of Life $\\uparrow$ & 5.069{\\scriptsize$\\pm$0.001} & 5.053{\\scriptsize$\\pm$0.000} & $-0.016${\\scriptsize$\\pm$0.000}$^{****}$ \\\\',
    ],
    'table5.6': [
        "Method & $n$ & Mean $\\pm$ SD & SE & 95\\% CI & $p$ vs.\\ GeoMACA & Cohen's $d$ \\\\",
        '\\textbf{GeoMACA} & 10 & 0.626 $\\pm$ 0.059 & 0.019 & [0.583,\\,0.668] & {--} & {--} \\\\',
        'COMA & 10 & 0.492 $\\pm$ 0.040 & 0.013 & [0.463,\\,0.520] & $<\\!10^{-4}$ & 2.66 \\\\',
        'QMIX & 10 & -0.302 $\\pm$ 0.029 & 0.009 & [-0.323,\\,-0.282] & $<\\!10^{-4}$ & 19.98 \\\\',
        'VDN & 10 & -0.046 $\\pm$ 0.044 & 0.014 & [-0.077,\\,-0.015] & $<\\!10^{-4}$ & 12.93 \\\\',
        'IPPO & 10 & 0.577 $\\pm$ 0.089 & 0.028 & [0.514,\\,0.641] & 0.1725 & 0.64 \\\\',
        'Uniform & 10 & 0.488 $\\pm$ 0.107 & 0.034 & [0.411,\\,0.565] & 0.0032 & 1.59 \\\\',
    ],
    'table5.7': [
        "Method & $n$ & Mean $\\pm$ SD & SE & 95\\% CI & $p$ vs.\\ GeoMACA & Cohen's $d$ \\\\",
        '\\textbf{GeoMACA} & 10 & 0.429 $\\pm$ 0.054 & 0.017 & [0.390,\\,0.468] & {--} & {--} \\\\',
        'COMA & 10 & 0.657 $\\pm$ 0.076 & 0.024 & [0.602,\\,0.711] & $<\\!10^{-4}$ & -3.44 \\\\',
        'QMIX & 10 & 0.030 $\\pm$ 0.034 & 0.011 & [0.006,\\,0.054] & $<\\!10^{-4}$ & 8.84 \\\\',
        'VDN & 10 & 0.001 $\\pm$ 0.002 & 0.000 & [-0.000,\\,0.002] & $<\\!10^{-4}$ & 11.17 \\\\',
        'IPPO & 10 & 0.394 $\\pm$ 0.088 & 0.028 & [0.330,\\,0.457] & 0.2957 & 0.48 \\\\',
        'Uniform & 10 & 0.485 $\\pm$ 0.071 & 0.023 & [0.434,\\,0.536] & 0.0649 & -0.88 \\\\',
    ],
}

CAPTIONS = {
    'table5.2': "\\caption{Unified-protocol CityBuilder evaluation. All methods use the same uniform-random developer at evaluation time. Heuristic has 5 evaluation replicates; learned methods have 10 retrained seeds. Cells are mean $\\pm$ SE across replicates. \\textbf{Bold} marks the best mean per metric; significance markers are Welch's two-sample $t$-tests against GeoMACA: $^*p<0.05$, $^{**}p<0.01$, $^{***}p<10^{-3}$, $^{****}p<10^{-4}$.}",
    'table5.3': '\\caption{Aggregated CityBuilder comparison. \\emph{Top-2} is the count of metrics on which the method ranks first or second among the four methods. \\emph{Avg.\\ rank} averages the per-metric rank (lower better). Pillar columns are z-scores within the four-method group; negative-direction metrics (Density, Segregation) are sign-flipped.}',
    'table5.4': "\\caption{GeoMACA component ablation ($n{=}7$ seeds per row). Each cell is mean $\\pm$ SE across seeds under the same unified uniform-developer evaluation protocol as Tab.~\\ref{tab:p3-main}; Full GeoMACA is the GeoMACA column of Tab.~\\ref{tab:p3-main} restricted to these seven seeds. Bold marks the best mean. The evaluated checkpoint comes from the warm-up for $3$, $1$, $1$, and $2$ of the seven seeds in the four rows. No ablation differs significantly from Full GeoMACA on any of the four metrics, even before Holm's correction (Welch's $t$-tests).}",
    'table5.5': '\\caption{CityBuilder without a government. The Heuristic and No government columns use the protocol of Tab.~\\ref{tab:p3-main} (uniform-random developer, $5$ replicates of $50$ episodes) and give mean $\\pm$ SE across replicates; the Heuristic column repeats Tab.~\\ref{tab:p3-main}. The last column is the mean difference, No government minus Heuristic, when both are run on the same developer plans and random numbers in every episode, $\\pm$ SE across the five replicates; paired $t$-tests: $^{****}p<10^{-4}$. Woodland and Density are identical in every paired episode.}',
    'table5.6': '\\caption{LBF final-training-window return: mean of the last $5{,}000$ training-episode returns per seed. Training returns include the proximity-shaping term (Sec.~\\ref{sec:p3-domain}); IPPO denotes independent PPO learners.}',
    'table5.7': '\\caption{LBF held-out evaluation. Each seed contributes the mean episodic return, food reward only, over 100 held-out evaluation episodes of its trained policy; IPPO denotes independent PPO learners.}',
}

FIGURE_5_3 = {
    'Economy': {'Heuristic': -0.30506482743912006, 'QMIX': -0.48248171465665335, 'COMA': -0.6079010870931086, 'GeoMACA': 1.3954476291888716},
    'Society': {'Heuristic': -0.26376703595560536, 'QMIX': -0.23706464607123648, 'COMA': 0.20772490089099363, 'GeoMACA': 0.29310678113586086},
    'Environment': {'Heuristic': -1.1262631230244964, 'QMIX': 0.05314490813309416, 'COMA': 1.1689462934547519, 'GeoMACA': -0.09582807856327379},
    'Overall': {'Heuristic': -0.5650316621397405, 'QMIX': -0.22213381753159855, 'COMA': 0.2562567024175456, 'GeoMACA': 0.5309087772538196},
}
