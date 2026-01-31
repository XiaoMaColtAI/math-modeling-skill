# 2026 MCMProblem C: Data With The Stars

![](images/4bc01e42be38c629fa388c4fe8c64a650dc2f0561d90ac656f9e789bcebaf7aa.jpg)

Dancing with the Stars (DWTS) is the American version of an international television franchisebased on the British show “Strictly Come Dancing” (“Come Dancing” originally). Versions ofthe show have appeared in Albania, Argentina, Australia, China, France, India, and many othercountries. The U.S. version, the focus of this problem, has completed 34 seasons.

Celebrities are partnered with professional dancers and then perform dances each week. A panelof expert judges scores each couple’s dance, and fans vote (by phone or online) for their favoritecouple that week. Fans can vote once or multiple times up to a limit announced each week.Further, fans vote for the star they wish to keep, but cannot vote to eliminate a star. The judgeand fan votes are combined in order to determine which couple to eliminate (the lowestcombined score) that week. Three (in some seasons more) couples reach the finals and in theweek of the finals the combined scores from fans and judges are used to rank them from $1 ^ { \mathrm { s t } }$ to $3 ^ { \mathrm { r d } }$(or 4th, 5th).

There are many possible methods of combining fan votes and judge scores. In the first twoseasons of the U.S. show, the combination was based on ranks. Season 2 concerns (due tocelebrity contestant Jerry Rice who was a finalist despite very low judge scores) led to amodification to use percentages instead of ranks. Examples of these two approaches are providedin the Appendix.

In season 27, another “controversy” occurred when celebrity contestant Bobby Bones wondespite consistently low judges scores. In response, starting in season 28 a slight modcation tothe elimination process was made. The bottom two contestants were identified using thecombined judge scores and fan votes, and then during the live show the judges voted to selectwhich of these two to eliminate. Around this same season, the producers also returned to usingthe method of ranks to combine judges scores with fan votes as in seasons one and two. Theexact season this change occurred is not known, but it is reasonable to assume it was season 28.

Judge scores are meant to reflect which dancers are technically better, although there is somesubjectivity in what makes a dance better. Fan votes are likely much more subjective, influencedby the quality of the dance, but also the popularity and charisma of the celebrity. Show producersmight actually prefer, to some extent, conflicts in opinions and votes as such occurrences boostfan interest and excitement.

Data with judges scores and contestant information is provided and described below. You maychoose to include additional information or other data at your discretion, but you mustcompletely document the sources. Use the data to:

Develop a mathematical model (or models) to produce estimated fan votes (which areunknown and a closely guarded secret) for each contestant for the weeks they competed.

o Does your model correctly estimate fan votes that lead to results consistent with whowas eliminated each week? Provide measures of the consistency.

o How much certainty is there in the fan vote totals you produced, and is that certaintyalways the same for each contestant/week? Provide measures of your certainty for theestimates.

• Use your fan vote estimates with the rest of the data to:

o Compare and contrast the results produced by the two approaches used by the show tocombine judge and fan votes (i.e. rank and percentage) across seasons (i.e. apply bothapproaches to each season). If differences in outcomes exist, does one method seemto favor fan votes more than the other?

o Examine the two voting methods applied to specific celebrities where there was“controversy”, meaning differences between judges and fans. Would the choice ofmethod to combine judge scores and fan votes have led to the same result for each ofthese contestants? How would including the additional approach of having judgeschoose which of the bottom two couples to eliminate each week impact the results?Some examples you might consider (there may also be others you identified):

 season 2 –Jerry Rice, runner up despite the lowest judges scores in 5 weeks.

 season 4 – Billy Ray Cyrus was $5 ^ { \mathrm { t h } }$ despite last place judge scores in 6 weeks.

 season 11 – Bristol Palin was $3 ^ { \mathrm { r d } }$ with the lowest judge scores 12 times.

 season 27 – Bobby Bones won the despite consistently low judges scores

o Based on your analysis, which of the two methods would you recommend using forfuture seasons and why? Would you suggest including the additional approach ofjudges choosing from the bottom two couples?

Use the data including your fan vote estimates to develop a model that analyzes the impact ofvarious pro dancers as well as characteristics for the celebrities available in the data (age,industry, etc). How much do such things impact how well a celebrity will do in thecompetition? Do they impact judges scores and fan votes in the same way?

Propose another system using fan votes and judge scores each week that you believe is more“fair” (or “better” in some other way such as making the show more exciting for the fans).Provide support for why your approach should be adopted by the show producers.

• Produce a report of no more than 25 pages with your findings and include a one- to two-pagememo summarizing your results with advice for producers of DWTS on the impact of howjudge and fan votes are combined with recommendations for how to do so in future seasons.

Your PDF solution of no more than 25 total pages should include:

• One-page Summary Sheet.

• Table of Contents.

• Your complete solution.

• One- to two-page memo.

• References list.

AI Use Report (If used does not count toward the 25-page limit.)

Note: There is no specific required minimum page length for a complete MCM submission. Youmay use up to 25 total pages for all your solution work and any additional information you wantto include (for example: drawings, diagrams, calculations, tables). Partial solutions are accepted.We permit the careful use of AI such as ChatGPT, although it is not necessary to create asolution to this problem. If you choose to utilize a generative AI, you must follow the COMAPAI use policy. This will result in an additional AI use report that you must add to the end of yourPDF solution file and does not count toward the 25 total page limit for your solution.

Data File: 2026_MCM_Problem_C_Data.csv – contestant information, results, and judgesscores by week for seasons 1 – 34. The data description is provided in Table 1.

Table 1: Data Description for 2026_MCM_Problem_C_Data.csv

<table><tr><td>Variables</td><td>Explanation</td><td>Example</td></tr><tr><td>celebrity_name</td><td>Name of celebrity contestant (Star)</td><td>Jerry Rice, Mark Cuban, ...</td></tr><tr><td>ballroompartner</td><td>Name of professional dancer partner</td><td>Cheryl Burke, Derek Hough, ...</td></tr><tr><td>celebrity_industry</td><td>Star profession category</td><td>Athlete, Model, ...</td></tr><tr><td>celebrity_homestate</td><td>Star home state (if from U.S.)</td><td>Ohio, Maine, ...</td></tr><tr><td>celebrity_homecountry/region</td><td>Star home country/region</td><td>United States, England, ...</td></tr><tr><td>celebrity_age during season</td><td>Age of the star in the season</td><td>32, 29, ...</td></tr><tr><td>season</td><td>Season of the show</td><td>1, 2, 3, ..., 32</td></tr><tr><td>results</td><td>Season results for the start</td><td>1st Place, Eliminated Week 2, ...</td></tr><tr><td>placement</td><td>Final place for the season (1 best)</td><td>1, 2, 3, ...</td></tr><tr><td>weekXjudgeY_score</td><td>Score from judge Y in week X</td><td>1, 2, 3, ...</td></tr></table>

# Notes on the data:

1. Judges scores for each dance are from 1 (low) to 10 (high).

a. In some weeks the score reported includes a decimal (e.g. 8.5) because each celebrityperformed more than one dance and the scores from each are averaged.

b. In some weeks, bonus points were awarded (dance offs etc); they are spread evenlyacross judge/dance scores.

c. Team dance scores were averaged with scores for each individual team member.

2. Judges are listed in the order they scored dances; thus “Judge Y” may not be the same judgefrom week to week, or season to season.
3. The number of celebrities is not the same across the seasons, nor is the number of weeks theshow ran.
4. Season 15 was the only season to feature an all-star cast of returning celebrities.
5. There are occasionally weeks when no celebrity was eliminated, and others where more thanone was eliminated.
6. N/A values occur in the data set for

a. the $4 ^ { t h }$ judge score if there is not $4 ^ { t h }$ judge for that week (usually there are 3) and$b$ . in weeks that the show did not run in a season (for example, season 1 lasted 6 weeksso N/A values are recorded for weeks 7 thru 11).

7. A 0 score is recorded for celebrities who are eliminated. For example, in Season 1 the firstcelebrity eliminated was Trista Sutter at the end of the Week 2 show. She thus has scores of 0for the rest of the season (week 3 through week 6).

# Appendix: Examples of Voting Schemes

# 1. COMBINED BY RANK (used in seasons 1, 2, and 28a - 34)

In seasons 1 and 2 judges and fan votes were combined by rank. For example, in season 1,week 4 there were four remaining contestants. Rachel Hunter was eliminated meaning shereceived the lowest combined rank. In Table 2 the judges scores and ranks are shown, and wecreated one possible set of fan votes that would produce the correct result. There are manypossible values for fan votes that would also give the same results. You should not use these asactual values as this is just one example. Since Rachel was ranked $2 ^ { \mathrm { n d } }$ by judges, in order tofinish with the lowest combined score, she has the lowest fan vote $4 ^ { \mathrm { t h } }$ place) for a total rank of 6.

Table 2: Example of Combining Judge and Fan Votes by Rank (Season 1, Week 4)

<table><tr><td>Contestant</td><td>Total Judges Score</td><td>Judges Score Rank</td><td>Fan Vote*</td><td>Fan Rank*</td><td>Sum of ranks</td></tr><tr><td>Rachel Hunter</td><td>25</td><td>2</td><td>1.1 million</td><td>4</td><td>6</td></tr><tr><td>Joey McIntyre</td><td>20</td><td>4</td><td>3.7 million</td><td>1</td><td>5</td></tr><tr><td>John O’Hurley</td><td>21</td><td>3</td><td>3.2 million</td><td>2</td><td>5</td></tr><tr><td>Kelly Monaco</td><td>26</td><td>1</td><td>2 million</td><td>3</td><td>4</td></tr></table>

* Fan vote/rank are unknown, hypothetical values chosen to produce the correct final ranks

# 2. COMBINED BY PERCENT (used for season 3 through 27a)

Starting in season 3 scores were combined using percents instead of ranks. An example isshown using week 9 of season 5. In that week, Jennie Garth was eliminated. Again, weartificially created fan votes that produce total percents to correctly lead to that result. Thejudges’ percent is computed by dividing the total judge score for the contestant by the sum oftotal judge scores for all 4 contestants. Based on the judges’ percent, Jennie was $3 ^ { \mathrm { r d } }$ . However,adding the percent of the 10 million artificially created fan votes we assigned to the judges’percent she was 4th. $4 ^ { \mathrm { t h } }$

Table 3: Example of Combining Judge and Fan Votes by Percent (Season 5, Week 9)

<table><tr><td>Contestant</td><td>Total Judges Score</td><td>Judges Score Percent</td><td>Fan Vote*</td><td>Fan Percent*</td><td>Sum of Percent</td></tr><tr><td>Jennie Garth</td><td>29</td><td>29/117 = 24.8%</td><td>1.1 million</td><td>1.1/10 = 11%</td><td>35.8</td></tr><tr><td>Marie Osmond</td><td>28</td><td>28/117 = 23.9%</td><td>3.7 million</td><td>3.7/10 = 37%</td><td>60.9</td></tr><tr><td>Mel B</td><td>30</td><td>30/117 = 25.6%</td><td>3.2 million</td><td>3.2/10 = 32%</td><td>57.8</td></tr><tr><td>Helio Castroneves</td><td>30</td><td>30/117 = 25.6%</td><td>2 million</td><td>2/10 = 20%</td><td>45.6</td></tr><tr><td>Total</td><td>117</td><td></td><td>10 million</td><td></td><td></td></tr></table>

* Fan vote is unknown, values hypothetical to produce the correct final standings

a The year of the return to the rank based method is not known for certain; season 28 is areasonable assumption.
