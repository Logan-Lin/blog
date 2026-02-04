+++
title = "Mini Project"
date = 2025-10-21
description = ""
weight = 28

[extra]
chapter = "Module B.8"
+++
Leveraging the knowledge from the [phase A and B](@/ai-system/_index.md) of this course (and optionally the advanced techniques from the phase C), we will develop and deploy a multi-functional AI API server in our mini project.

## Outcome

The outcome of the project is an AI API server (similar to the one we implemented in [module 3](@/ai-system/wrap-ai-with-api/index.md)), deployed on a machine other than your personal PC. A client program that can interact with the server should also be included.

### Necessary Requirements

#### Implementation of the API Server

The API server should have more than one route (a.k.a. API endpoint). You have the flexibility to plan the functionality of these routes. Note that at least one of them should incorporate "AI functionality", either image or language-related. Examples include:

- One route `<domain>/v1/image_classify` for image classification and another route `<domain>/v1/conversation` for LLM-powered natural language conversation
- Multiple versions `<domain>/v1/image_classify` and `<domain>/v2/image_classify` providing different sets of functionality
- Utility routes like `<domain>/v1/model` for listing available AI models

For API framework, although we have only learned FastAPI in this course, you have the freedom to use other frameworks (e.g., Flask, Django, or even programming languages other than Python) if you want.

For the AI model powering the AI functionality, you can totally use off-the-shelf models from HuggingFace, the model you prepared for other courses, or elsewhere; this is not the focus of this mini project.

You are free to use libraries, ask AI for help, or even reference to the code I included in the blog post. But you shouldn't directly copy and paste any code (implemented by others or AI) without any modification, especially if you have no idea of the meaning of the code. In other words, I am not against reuse of existing tools and code since it is a common practice in software development, but you have to ensure that you understand your implementation.

#### Deployment of the API Server

In general, you can deploy the API server on any machine as long as it is not the same machine the client program is going to run on. The actual purpose of this requirement is the client program and the server should live in different host environments so the knowledge of interaction through network is required. In practice you can interpret this requirement quite flexibly, below is a list of examples that are all accepted:
- You run the server on your personal computer, and run the client on your colleague/friend's personal computer
- You have two personal computers, with one of them running the server and another one running the client
- You run the server on a Raspberry Pi/NVIDIA Jetson, and run the client on your personal computer, or reversely
- You run the server on a cloud computer, and run the client on your personal computer, or reversely
- You run both server and client on the same physical machine, but they are in different host environments: one of them is in a virtual machine or each is in a different virtual machine

The server should be deployed using containerization technique we learned in [module 5](@/ai-system/packaging-containerization/index.md). In other words, you shouldn't run the server program directly on the host environment of the machine. You have the freedom to use container frameworks other than Docker (e.g., Podman).

#### Client Program and Interaction with the Server

You also should prepare a simple client program for validating that the server is functioning correctly. There is no strict requirement for any aspect of this client program, as long as it can demonstrate the functionality of the API server you deployed.

### Tips

These are not strict requirements, but are aspects that you probably should consider to demonstrate your understanding of the knowledge covered in this course:
- API endpoints design that adhere to the REST principle
- API versioning considerations, even if you plan to only have `v1` endpoints
- Integration with databases in the API server for API key management
- Leveraging the dedicated AI computing hardwares if the machine has any
- Build the server container image with Dockerfile and proper layering

### Optional Achievements

For folks who feel the above requirements are too easy, here are some examples of how you can incorporate advanced techniques into your mini project to amaze your classmates. You probably have even crazier ideas if you are really considering implementing these: the sky is the limit here.
- Record per user usage and implement advanced rate limit algorithm for your API endpoints with AI functionalities
- Leverage production-ready techniques we are going to learn in the last phase to: make your server highly-available (e.g., run on a cluster) or implement advanced deployment strategies
- Make your server publicly available, by giving it a publicly accessible IP address, a domain, and proper SSL certificate
- Make your API server a drop-in replacement of OpenAI/Anthropic's API families by implementing multi-modal conversational APIs that can receive both natural language and image input and generate language output

I do want to note that, to keep it fair, incorporating these optional achievements in the project does not directly grant you a higher score than those who don't. The purpose of the project is to reflect your understanding of the knowledge we covered in this course, and the course is scored based on the oral exam, not the project outcome directly.

## Report

As you probably already know, you need to submit a report for this mini project. The format of the report is as follows.

3-8 pages, excluding references, containing:
- Title and all authors
- Introduction: a short problem analysis
- Implementation: Explain important design and implementation choices of the API server and the client program
- Deployment: Demonstrate the important steps of deploying the API server
- Results: Evaluation of the API server's functionality and your reflections
- Conclusion

> **Tip:** The report can provide a concise explanation of your key design choices, the rationale behind them, and the deployment process you followed. Focus on demonstrating your understanding of the system you built and showing that it achieves its intended objectives. Think of it as walking the reader through your thought process and the outcomes of your work, rather than providing exhaustive code documentation.

## Submission

In the end, you should submit one `.zip` or `.tar.gz` (or other open file bundle formats) file containing:
- The report in PDF format
- All source code necessary to build the API server container, including the source code of the API server, the `Dockerfile`, among others if needed (e.g., `requirements.txt`)
- Source code of the client program

**One submission per group** should be uploaded in DigitalExam no later than: December 4th 23:59, 2025 (Copenhagen time).
