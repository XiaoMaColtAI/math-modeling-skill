# 2026 MCM

# Problem A: Modeling Smartphone Battery Drain

![](images/2b5d356fc542fea6f13c92bb8588b00f0a8255033472c6e0203c1fd673ff1541.jpg)



25%


![](images/9ec42ac02eb37acd1f63743c114bad0fcbcb739f6e35dde2e852dcc98423bdd9.jpg)



50%


![](images/c3baaf0ddbb7528ec969bc9227d45634e34eff9369df76283e9126574c54565e.jpg)



75%


![](images/7c3765aec34f9e2c3dd62910497dfdf5184b7ed353bc82155c9d5938a421a03d.jpg)



100%


Smartphones are indispensable tools in modern life, yet their battery behavior often seems unpredictable. On some days a phone may last the whole day; on other days it drains rapidly before lunch. Although some users attribute this to "heavy use," the true drivers of battery depletion are more complex. Power consumption depends on the interplay of screen size and brightness, processor load, network activity, and background applications that continue drawing energy even when the device appears idle. Environmental conditions such as temperature further complicate matters: some batteries lose effective capacity in cold weather and may overheat under sustained heavy use. A battery's behavior is also influenced by its history and how it has been charged during its lifetime.

Your task is to develop a continuous-time mathematical model of a smartphone's battery that returns the state of charge (SOC) as a function of time under realistic usage conditions. This will be used to predict the remaining time-to-empty under different conditions. You should assume that the phone has a lithium-ion battery.

# Requirements:

1. Continuous-Time Model: Develop a model to represent the state of charge using a continuous-time equation or system of equations. You may want to begin with the simplest reasonable description of battery drain and then extend it to incorporate additional contributors such as screen usage, processor load, network connections, GPS usage, and other background tasks.

Data as support, not substitute: You may collect or use data for parameter estimation and validation. If open datasets are limited, you may use published measurements or specifications (with proper citation), provided parameters are clearly justified and validated for plausibility. However, projects based solely on discrete curve fitting, timestep regression, or black-box machine learning without an explicit continuous-time model will not satisfy this problem's requirements. All data used must be well documented and freely available, and the data must be free for use under an open license.

2. Time-to-Empty predictions: Use your model to compute or approximate the time-to-empty under various initial charge levels and usage scenarios. Compare predictions to observed or plausible behavior, quantify uncertainty, and identify where the model performs well or poorly.

○ Show how your model explains differences in these outcomes and identify the specific drivers of rapid battery drain in each case.

- Which activities or conditions produce the greatest reductions in battery life? Which ones change the model surprisingly little?

3. Sensitivity and Assumptions: Examine how your predictions vary after making changes in your modeling assumptions, parameter values, and fluctuations in usage patterns.

4. Recommendations: Translate your findings into practical recommendations for a cellphone user. For example, which user behaviors—such as reducing brightness, disabling background tasks, or switching network modes—yield the largest improvements in battery life? How might an operating system implement more effective power-saving strategies based on insights from your model? Consider how battery aging reduces effective capacity or how your modeling framework could generalize to other portable devices.

# Your report should present:

- A clear description of your model and governing equations.

- The assumptions and rationale behind your design choices.

Parameter estimation methods and validation results.

- A discussion of strengths, limitations, and possible extensions.

- An executive-style summary highlighting main results, insights, and recommendations.

Important: Your model must be grounded in clearly defined physical or mechanical reasoning; discrete curve fitting or other mathematical forms that are disconnected from an explicit continuous-time description of battery behavior will not satisfy the requirements. Projects that rely solely on discrete curve fitting or statistical regression without a clearly formulated continuous-time model will not satisfy the requirements of this problem.

Your PDF solution of no more than 25 total pages should include:

One-page Summary Sheet.

Table of Contents.

- Your complete solution.

In-text Citations and A Reference List.

AI Use Report (If used does not count toward the 25-page limit.)

Note: There is no specific required minimum page length for a complete MCM submission. You may use up to 25 total pages for all your solution work and any additional information you want to include (for example: drawings, diagrams, calculations, tables). Partial solutions are accepted. We permit the careful use of AI such as ChatGPT, although it is not necessary to create a solution to this problem. If you choose to utilize a generative AI, you must follow the COMAP AI use policy. This will result in an additional AI use report that you must add to the end of your PDF solution file and does not count toward the 25 total page limit for your solution.

# Glossary

Smartphone: is a mobile device that combines the functionality of a traditional cell phone with advanced computing capabilities.

Power Consumption: the rate at which a device uses electrical energy from its battery or power source.

Processor Load: the actual amount of work being done by the processor at a given moment.

State of Charge (SOC): a measure of how much energy remains in a battery compared to its full capacity, expressed as a percentage.

Time-to-Empty: the estimated amount of time remaining before a battery is completely discharged.