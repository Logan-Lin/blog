+++
title = "Interact with AI Systems"
date = 2025-09-03
description = ""
weight = 10

[extra]
chapter = "Phase A"
+++
> **TL;DR:**
> Learn why standardized interactions between applications are essential for making AI models practical in real-world scenarios, moving beyond simple function calls to robust communication methods that work across different programming languages and distributed systems.

Interactions are common in both human society and the digital world. We as humans interact with each other through language so that our thoughts and purposes are communicated. In the digital world, a food delivery application interacts with restaurants to process your order and interacts with banks to ensure payment for the meal. Interactions between applications (AI or not) are what build the digital world we cannot live without. But how do applications interact with each other?

From the previous course [DAKI2 - Design og udvikling af AI-systemer](https://www.moodle.aau.dk/course/view.php?id=50253) you already know how an AI model is designed and implemented. You will even dive deeper with the parallel course [DAKI3 - Deep learning](https://www.moodle.aau.dk/course/view.php?id=57017). To some extent, you have already witnessed interactions between applications: when you feed data into the `forward` function of your AI model and retrieve its output data to perform visualization, different components of your software are interacting with each other.

> **Videos:**
> - [How Python function calls work internally](https://www.youtube.com/watch?v=nXLhpuKI4Mw)

But this form of interaction is not suitable in many real-world scenarios, as soon as our applications grow beyond a single piece of software. For example, AI models nowadays are largely written in Python, but not all softwares are. Needless to say, it is quite impractical to make software written in Rust call a Python function. Even if Python were the one and only programming language in the universe, we cannot expect our digital world to be one giant Python project where applications are functions that can call each other.

In conclusion, to make our AI models practical in real-world use cases, we need a more standardized and streamlined means of letting them interact with the outside world, which is the subject of this phase of the course. Note that for now we will be focusing on existing AI models/systems and locally run AI models. We will leave the deployment of AI models for later phases of the course.
