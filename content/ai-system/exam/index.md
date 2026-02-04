+++
title = "Exam Format"
date = 2025-09-02
description = ""
weight = 50

[extra]
chapter = ""
+++
## Modality and duration

Individual oral exam based on submitted project. The duration will be 15 minutes, followed by 5 minutes deliberation.

### Agenda

- Students give a (roughly) 5-minute presentation of their completed mini-project
- Students randomly draw a topic (from the 6 topics listed below) and explain the basic concepts within that topic
- Examiner and censor ask follow-up questions, which may relate to other course topics and include questions about practical applications and implementation; students can optionally refer back to their mini-project

## Assessment

Students are not required to write code during the exam, nor to remember any specific commands or code syntax, but may be asked to draw diagrams or solve small tasks manually. The grade will be based on an overall assessment of both the mini-project and oral performance, and in accordance with the 7-point grading scale.

## Pre-approved aids

Notes (related to literature, slides, notes from the module, and project documentation) are permitted. Do note that if you read directly from notes or copy them verbatim, you may be asked to put the notes away. Answers based solely on reading from notes will result in failure.

## Prerequisites for participation

Timely hand-in of project documentation.

## Topics

Note that the questions listed below are examples and may be formulated differently during the exam. An exhaustive explanation of each topic is not expected. While ideally students should be able to discuss each topic covered in the course, this is not required to pass the exam. The number and formulation of questions can serve as an indicator of the importance and expected level of knowledge for each topic.

### 1. Interacting with APIs

- What is the primary purpose of APIs in software development? How do they enable standardized communication between applications?
- Explain the three pillars of APIs: network fundamentals (IP addresses, domains, ports), HTTP protocol & methods, and standards & design principles
- Walk through the components of an HTTP request (request line, headers, body) and response (status line, headers, body).
- What are the key request headers required for API authentication and content specification?
- Explain the difference between GET and POST methods. When would you use each for AI API interactions?
- What are the key components of interacting with APIs using Python's requests library? Explain the principles of proper error handling and API key management
- What is rate limiting and why is it important for AI APIs? Compare different rate limiting strategies
- Compare traditional request-response APIs and streaming APIs. Why is streaming preferred for conversational AI applications?

### 2. Building APIs

- Explain the concept of routes in FastAPI and how GET and POST methods are used.
- What are URL parameters and how do they enable dynamic request handling?
- Explain the role of Pydantic data models in FastAPI for request/response validation. Why is this important for API reliability?
- Explain how API versioning can be implemented in FastAPI. Why is maintaining backwards compatibility important?
- What are the key considerations when integrating AI models (like image classification) into FastAPI servers? Why might asynchronous operations be important?
- What are the principles of implementing API key authentication in FastAPI? What are the security considerations?
- Explain why database integration (using SQLAlchemy) is important for user management and API usage tracking.

### 3. Computing architecture & hardware

- Computer architecture fundamentals
    - Explain the Von Neumann architecture and its main components. How do modern computers (including "AI computers") relate to this 80-year-old architecture?
    - What is the difference between instructions and data in a computer system? Why does the Von Neumann architecture store both in the same memory?
    - Explain the roles of the Control Unit (CU) and Arithmetic Logic Unit (ALU) in a CPU. Use an analogy to illustrate their relationship
    - Describe the role of bus systems in computer architecture. What are the three main types of buses and what does each carry?
- AI computing hardware
    - Why are CPUs designed for sequential processing? What makes this approach less suitable for AI workloads?
    - Explain the difference between sequential and parallel processing using an analogy. Why do AI models (especially neural networks) benefit from parallel processing?
    - What is the memory bus bottleneck for AI workloads? Why is memory bandwidth more critical than latency for AI computing?
    - Why are GPUs particularly well-suited for AI computing? Explain in terms of core architecture and memory design
    - Compare GPUs, TPUs, and NPUs in terms of their design goals, strengths, and typical use cases
    - How does specialized AI hardware (GPU, TPU, NPU) relate to the Von Neumann architecture at the system level? Does it fundamentally replace the Von Neumann architecture?
    - What factors should you consider when choosing hardware for different AI applications (training vs inference, data center vs edge device)?

### 4. Containerization

- Container fundamentals
    - What deployment problem do containers solve? Explain the "it works on my machine" syndrome and how containerization addresses it
    - What is a container and how does it achieve isolation? Explain the benefits of containers over traditional deployment approaches
    - Describe the layered structure of container images using an analogy. How does this contribute to efficiency and reusability?
    - What is a Dockerfile and why is it the preferred approach for building container images? How does it relate to the layered structure and reproducibility?
    - What are the main components of the Docker ecosystem? Explain the roles of Docker Engine, CLI, Dockerfile, and registries like Docker Hub
    - What is the purpose of container registries, and how do they enable image distribution? Compare different registry options
- Practical implementation
    - Explain how containers handle port mapping. Why is this important for deploying web applications and API servers?
    - How do you manage persistent storage in containerized applications? Why is this important for AI applications with databases or model files?
    - How should you handle configuration in containerized applications to avoid hardcoding them in the code?  What approaches are available?
    - How does the order of instructions in a Dockerfile relate to the layered structure of images? What are the important layers you will consider when containerizing an AI API server?
    - What is the difference between data stored in container image layers versus data stored in volumes? Why does this distinction matter for large files like AI models or databases?

### 5. Cloud deployment

- What is cloud computing and how does virtualization enable cloud infrastructure?
- Compare virtual machines, container services, GPU instances, and managed AI services. What are the trade-offs?
- What are the main steps and considerations when deploying a containerized AI service on a cloud virtual machine?
- How would you choose between different cloud providers for your AI deployment? Consider factors like pricing models, service offerings, and vendor lock-in
- What are the advantages and pitfalls of usage-based pricing vs fixed monthly pricing for cloud services?
- Why is HTTPS necessary for production AI APIs? Explain the user experience and security implications
- How do you obtain a domain name for your service? Compare free and paid options

### 6. Edge and self-hosted deployment

- What is edge computing and what motivates edge deployment for AI applications? Discuss latency, privacy, bandwidth, and offline operation
- What is self-hosted deployment and how does it differ from cloud deployment? Discuss the economics, control, and practical considerations
- Compare different hardware options for edge/self-hosted deployment: Raspberry Pi, NVIDIA Jetson, repurposed laptops, and purpose-built servers
- What are CPU architecture considerations when deploying containers to edge devices? Explain the ARM vs x86 challenge
- How do you access services on your local network vs making them publicly accessible? What is NAT and why does it matter?
- What are the different approaches for making self-hosted services publicly accessible? What are the trade-offs to consider when choosing an approach?
