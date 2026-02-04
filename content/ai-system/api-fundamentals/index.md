+++
title = "API Fundamentals"
date = 2025-09-03
description = ""
weight = 11

[extra]
chapter = "A.1"
+++

> **TL;DR:**
> APIs are standardized interfaces that enable applications to communicate across different programming languages and infrastructure, serving as a universal postal system for the digital world.

In [Interact with AI Systems](@/ai-system/interact-with-ai-systems/index.md) we've established that we need a standardized interaction method beyond direct function calls. That method is what we call **Application Programming Interfaces (APIs)**. If applications need to communicate like humans do but face barriers like different programming languages and different deployment infrastructure, APIs are like having a universal postal office who knows where everyone lives and how they prefer to receive and send messages.

> **Example:**
> ChatGPT can be accessed through OpenAI's official website, mobile/desktop apps, other AI-based applications (such as Perplexity), Python scripts, or even command line scripts, all through the same family of APIs OpenAI has published.

![API overview](api-overview.webp)

## The Three Pillars of APIs

When humans communicate through letters, three pillars are needed: where to send the letters (recipient's address), how to send the letters (postal services and delivery methods), and the letter itself (format and content of the message). Similarly, APIs also need three pillars to work: where to send the message (network fundamentals), how to send the message (HTTP protocol & methods), and a "common knowledge" of how the APIs should be designed and used (standards & design principles).

### Network Fundamentals

Just like you need an address to send a letter, APIs need addresses too. Without going too deep into computer networking, we will focus on three core concepts: IP addresses, domains, and ports.

An **[IP address](https://www.geeksforgeeks.org/computer-science-fundamentals/what-is-an-ip-address/)** is a unique identifier assigned to each device connected to a network, telling applications where to find each other. Think of it as a street address such as *Fredrik Bajers Vej 7K, 9220 Aalborg East, Denmark*. An IPv4 address looks something like `65.108.210.169`.

Technically speaking, APIs can identify themselves solely with IP addresses. The problem is that IP addresses are difficult for humans to read and remember, just like street addresses are usually too long for us to remember. We usually prefer a shorter, semantic-rich name like *Aalborg University*. Similarly, domain names provide this human-friendly alternative. A **[domain](https://www.geeksforgeeks.org/computer-networks/introduction-to-domain-name/)** is also a unique identifier pointing to some network resource and usually has one (or more) corresponding IP address(es). In the ChatGPT example above, `api.openai.com` is the domain name of the API, pointing to IP addresses like `162.159.140.245` and `172.66.0.243`.

Finally, we have ports. Just as some people run several businesses in the same location and have multiple corresponding mailboxes, computers run multiple applications simultaneously. A **[port](https://www.geeksforgeeks.org/computer-networks/what-is-ports-in-networking/)** is used to identify which specific application should receive the incoming message, and each IP address can have up to 65,535 ports. Typically we don't have to specify a port when calling an API, since there are default ports assigned to certain services, protocols, and applications. For example, HTTPS-based APIs usually run on port 443.

We should also briefly address the [difference between a URL and a domain](https://www.geeksforgeeks.org/computer-networks/difference-between-domain-name-and-url/) here. Think of the domain `api.openai.com` as the building address like *Fredrik Bajers Vej 7K* that usually corresponds to a certain group of hardware resources. The full URL is like an address with floor and room number like *Fredrik Bajers Vej 7K, 3.2.50*, which in the below example specifies the version of the API (v1) and the specific function (conversation completion).

![URL structure](url-structure.webp)

> **Videos:**
> - [The OSI model of computer networks](https://www.youtube.com/watch?v=keeqnciDVOo)
> - [IP address explained](https://www.youtube.com/watch?v=7_-qWlvQQtY)
> - [How domains are mapped to IP addresses](https://www.youtube.com/watch?v=mpQZVYPuDGU)
> - [Network ports explained](https://www.youtube.com/watch?v=h5vq9hFROEA)
> - [Understanding URLs](https://www.youtube.com/watch?v=5Jr-_Za5yQM)

### HTTP Protocol & Methods

To send a letter in the real world, you first have to choose from available postal services, which you will probably choose based on price, delivery time, previous experiences, etc. For APIs, you usually won't spend time choosing postal services (transfer protocols) since they are largely standardized, and that one standard protocol used in most APIs is called **[HTTP (HyperText Transfer Protocol)](https://www.geeksforgeeks.org/html/what-is-http/)**.

What you do have to choose is **HTTP methods**, similar to how a postal service usually has multiple delivery methods. Two methods that you will frequently encounter when using AI service APIs are `GET` and `POST`. `GET` means the API call wants to retrieve information, for example you can check OpenAI's available AI models by sending a `GET` request to `https://api.openai.com/v1/models`. `POST` is for sending data and expecting a response, which will be the primary method we use to send data to AI services and retrieve their response.

#### HTTP Request

Besides providing multiple methods, HTTP as a postal service for APIs also standardize how each envelope is addressed, in the form of several [HTTP request components](https://proxyelite.info/understanding-http-requests-what-are-they-made-of/): request line, headers, and body.

The **request line** will be something like this:
```
POST https://api.openai.com/v1/chat/completions HTTP/1.1
```
This contains the method, the URL stating where to send the request, and the protocol version.

The **[headers](https://en.wikipedia.org/wiki/List_of_HTTP_header_fields)** are like the information you write on the envelope, and will be something like this:
```
Authorization: Bearer sk-abc1234567890qwerty
Content-Type: application/json
Accept: application/json
User-Agent: SomeAIApp/1.0
```
Here, `Authorization` is for identifying the user and protecting the API and is usually where we specify our API keys. `Content-Type` and `Accept` specify the format of data we're sending and the expected response, respectively. `User-Agent` identifies the type of application or client we are using to interact with the API.

For the `GET` method, only the request line and headers, or sometimes just the request line, is enough. For the `POST` method, since we are sending data, we need the **body** which is the content of the letter itself. As you noticed, in the headers we've stated that the format of the body will be `application/json`, which means our body will look like this:
```json
{
    "model": "gpt-4",
    "messages": [
        {"role": "user", "content": "Write a haiku about APIs"}
    ],
    "temperature": 0.7,
    "max_tokens": 50
}
```
The format of this JSON object is specified by the provider of the APIs. There are other [content types](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Content-Type) that might be more suitable for certain types of data. Generally speaking, JSON is the most popular one since it's machine-parseable and human-friendly.

#### HTTP Response

Now you've sent the envelope (HTTP request) through the postal service called HTTP. The recipient will send a response letter back to you (HTTP response) if everything is working correctly, and if not, the postal service will at least write a response telling you what's wrong. Akin to HTTP request, a [HTTP response](https://www.tutorialspoint.com/http/http_responses.htm) is composed of a few components: status line, response headers, and response body.

The **status line** looks like this:
```
HTTP/1.1 200 OK
```
Composed of HTTP protocol version, status code, and reason phrase. Both status code and reason phrase provide immediate information about how your sent request went, and they correspond one-to-one.

The **response headers** are like headers in the request, providing metadata about the response. It might look something like this:
```
Content-Type: application/json
Content-Length: 1247
```
The types of headers included in a response depend on the design of the API service and are largely relevant to the purpose of the API. For example, ChatGPT's API will provide information about the AI model and your current usage in their response headers.

The **response body** is similar to the body in the request, containing the data the API provider sends back to you. A response body from the ChatGPT API with JSON format will look like this:
```json
{
  "id": "chatcmpl-6pHh8Cw1ZKcO45PiAavgbhZMz3YRs",
  "object": "chat.completion",
  "created": 1677649420,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 13,
    "total_tokens": 25
  }
}
```
Again, the format of this JSON object is specific to API providers and functions you requested.

> **Videos:**
> - [HTTP explained](https://www.youtube.com/watch?v=KvGi-UDfy00)
> - [HTTP request explained](https://www.youtube.com/watch?v=DBhEFG7zjFU)

> **Note:**
> You might have noticed that we've been saying HTTP protocol throughout the above section, but the URLs we are calling start with HTTPS. HTTPS is an extension of HTTP that additionally encrypts messages. Think of it as writing letters in a way that only you and the recipient can understand. Nowadays, almost all public APIs use HTTPS and most software blocks all non-secure HTTP communications. We will come back to HTTP and HTTPS when we are deploying our own APIs in Module 6: [Cloud Deployment](@/ai-system/cloud-deployment/index.md).

### Standards & Design Principles

In communications, beyond mandating rules (e.g., languages) we have "common knowledge"—for example, how an address is written (street and building number, then post code and city/area, finally country) and how a letter is structured (greetings and regards). You can technically refuse to adhere to such common knowledge, but it might lead to miscommunication and confusion, or you will need to attach a document stating how and why you do things differently. Similarly, when working with APIs, there are standards and design principles that are not mandatory but will make the APIs more predictable and intuitive, reducing the need for users and developers to extensively study the API documentation.

We'll briefly touch on one of the more prominent and widely adopted standards: **[REST (Representational State Transfer)](https://amplication.com/blog/rest-apis-what-why-and-how)**. Core REST principles include uniform interface, statelessness, cacheability, and layered system.

**Uniform interface** ensures all interactions between applications follow a consistent pattern, for example, making the formulation of URLs intuitive and HTTP methods consistent. API URLs that follow this principle include:
```
GET    /v1/models                   # Get all models
GET    /v1/models/gpt-4             # Get specific model
POST   /v1/chat/completions         # Create a chat completion
GET    /v1/files                    # List uploaded files
POST   /v1/files                    # Upload a new file
```
And bad examples include:
```
POST   /getModels                   # Action in URL
GET    /model?action=delete&id=123  # Action as parameter
POST   /api?method=chat             # Generic endpoint
```

**Statelessness** requires that each request contain all information necessary to understand and process the request. One example is that OpenAI's chat completion API always requires the full chat history to be provided in the body:
```json
{
    "model": "gpt-4",
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
}
```

**Cacheability** means HTTP responses should clearly define themselves as cacheable or non-cacheable. This can make the communication and computation of applications more efficient. Especially for AI APIs, frequently requested AI outputs can be flagged as cacheable and don't need to be recalculated.

**Layered system** allows the architecture to be composed of multiple hierarchical layers, where each layer has specific roles and cannot see beyond the immediate layer it's communicating with. Typical AI APIs will include authentication layers for security, caching layers for reuse of frequently accessed AI results, and rate limiting layers to prevent abuse.

> **Videos:**
> - [What is a REST API?](https://www.youtube.com/watch?v=lsMQRaeKNDk)

## Interact with APIs in Practice

Now we've established the basic concepts related to APIs, we will look at how to interact with APIs in practice.

### API Testing Tools

Before we proceed to integrate interactions with APIs into our applications, we can play around with the APIs with API testing tools to first get a better idea of the behavior of the APIs. These tools will also come in handy when we implement our own APIs and want to test them ourselves before publishing them to the public.

[Postman](https://www.postman.com/) is a popular API testing tool. To send an API request with Postman, fill in the components of an [HTTP Request](#http-request) into its interface:

![Postman request](postman-request.webp)

Click send, and after a while you should be able to see the response with components of an [HTTP Response](#http-response):

![Postman response](postman-response.webp)

Feel free to explore other functionalities of Postman yourself. Apart from being able to send API requests in a graphical user interface, you can also form a collection of requests for reuse and structured testing. Postman also comes with collaboration tools that can come in handy when developing in a team. Alternatives to Postman include [Hoppscotch](https://hoppscotch.io/) and [Insomnia](https://insomnia.rest/), [among others](https://apisyouwonthate.com/blog/http-clients-alternatives-to-postman/), all with similar core functionalities.

### Interact with APIs with Python

To interact with APIs in a Python program, a universal method is to use the [`requests` package](https://docs.python-requests.org/en/latest/index.html). It is not a built-in package and you will have to install it with a package manager of your choice.

#### Sending `GET` Request

Below is an example of sending a `GET` request:

```python
import os
import requests

url = "https://api.anthropic.com/v1/messages"

headers = {
    "x-api-key": os.getenv("API_KEY"),
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "SomeAIApp/1.0",
    "anthropic-version": "2023-06-01"
}

try:
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    print(f"Response Body: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"GET request failed: {e}")
```

Let's break down each part and see how it connects to the concepts covered earlier.

The URL:

```python
url = "https://api.anthropic.com/v1/messages"
```

This maps directly to [Network Fundamentals](#network-fundamentals). The domain `api.anthropic.com` identifies where the API server is located, and the path `/v1/messages` specifies which specific endpoint (function) we want to access. This is like addressing a letter to a specific department in a building.

The headers:

```python
headers = {
    "x-api-key": os.getenv("API_KEY"),
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "SomeAIApp/1.0",
    "anthropic-version": "2023-06-01"
}
```

These are the [HTTP Request](#http-request) headers, metadata about the request. The `x-api-key` handles authorization (proving who you are), `Content-Type` and `Accept` specify we're working with JSON format, `User-Agent` identifies our application, and `anthropic-version` specifies the API version. Note the security best practice: using `os.getenv("API_KEY")` to retrieve the API key from environment variables rather than hardcoding it in your code.

The response:

```python
response = requests.get(url, headers=headers)
print(f"Status Code: {response.status_code}")
print(f"Response Headers: {response.headers}")
print(f"Response Body: {response.text}")
```

The response object contains all three components of an [HTTP Response](#http-response): `response.status_code` gives us the status code (e.g., 200 means success), `response.headers` provides the response headers with metadata about the response, and `response.text` contains the response body with the actual data the API returned.

#### Sending `POST` Request

And an example of sending a `POST` request:

```python
import os
import requests
import json

url = "https://api.anthropic.com/v1/messages"

headers = {
    "x-api-key": os.getenv("API_KEY"),
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "SomeAIApp/1.0",
    "anthropic-version": "2023-06-01"
}

json_body = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 2048,
    "temperature": 0.7,
    "messages": [
        {
            "role": "user",
            "content": "Explain the concept of APIs."
        }
    ]
}

try:
    response = requests.post(
        url,
        headers=headers,
        json=json_body,
        timeout=30  # 30 second timeout
    )

    response.raise_for_status()  # Raises HTTPError for bad responses

    result = response.json()
    print("Success!")
    print(f"Content: {result.get('content', [{}])[0].get('text', 'No content')}")

except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error occurred: {e}")
    print(f"Response content: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except json.JSONDecodeError:
    print("Failed to decode JSON response")
```

Let's break down each part to understand how `POST` requests differ from `GET` requests and how they map to HTTP concepts.

The request body:

```python
json_body = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 2048,
    "temperature": 0.7,
    "messages": [
        {
            "role": "user",
            "content": "Explain the concept of APIs."
        }
    ]
}
```

This is the [HTTP Request](#http-request) body, the actual data we're sending to the API. Unlike `GET` requests which only have headers, `POST` requests include a body with the information needed to process the request. Notice how we include the full `messages` array, following the [statelessness principle](#standards-design-principles) where each request contains all necessary information.

Sending the request:

```python
response = requests.post(
    url,
    headers=headers,
    json=json_body,
    timeout=30
)
```

The `requests.post()` function combines all the [HTTP Request](#http-request) components: the URL specifies where to send it, `headers` provides the metadata, and `json=json_body` automatically converts our Python dictionary to JSON format and sets it as the request body. The `timeout` parameter ensures we don't wait forever if something goes wrong.

Response handling and error management:

```python
response.raise_for_status()
result = response.json()
print(f"Content: {result.get('content', [{}])[0].get('text', 'No content')}")
```

The `raise_for_status()` method checks the [HTTP Response](#http-response) status code and raises an exception for error codes (4xx or 5xx). The `response.json()` parses the response body from JSON format into a Python dictionary, making it easy to extract specific fields.

Different error types:

```python
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error occurred: {e}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except json.JSONDecodeError:
    print("Failed to decode JSON response")
```

Different exceptions handle different failure scenarios: `Timeout` for when requests take too long, `HTTPError` for bad [HTTP Response](#http-response) status codes (caught by `raise_for_status()`), `RequestException` for general network problems, and `JSONDecodeError` for malformed response bodies. This demonstrates robust error handling practices for API interactions.

> **Videos:**
> - [Python `requests` library tutorial](https://www.youtube.com/watch?v=tb8gHvYlCFs)

> **Extended Reading:**
> To get started with AI APIs, you'll need to register accounts, obtain API keys, and familiarize yourself with provider documentation. Here are the two major AI API platforms to explore:
>
> - [OpenAI platform](https://platform.openai.com/welcome)
> - [Anthropic developer console](https://console.anthropic.com/)
>
> And their documentation:
> - [OpenAI API Documentation](https://platform.openai.com/docs/overview)
> - [Anthropic API Documentation](https://docs.anthropic.com/en/api/overview)

## Exercise

Build two Python programs that demonstrate the API fundamentals covered in this module.

**Exercise 1: Command-line Chatbot**

Build a chatbot using an AI API of your choice (e.g., OpenAI, Anthropic, or others) that takes users' input from the command line interface and display the response. It demonstrates:
- **HTTP Request Components**: Properly implement headers including Authorization, Content-Type, and User-Agent as shown in the [HTTP Request](#http-request) section
- **REST Statelessness**: Follow the statelessness principle by including full conversation history in each request
- **HTTP Status Code Handling**: Handle different status codes with user-friendly messages referencing the [HTTP Response](#http-response) section
- **Response Processing**: Parse and display relevant response (content, usage tokens, model information)

**Exercise 2: Image Analysis Tool**

Build a command-line tool that analyzes image content using a multi-modal AI API of your choice that takes users' input file (path) from the command line and display the analysis result (content of the image, class labels, or others). It demonstrates the same points as above plus:
- **Content-Type Handling**: Choose a proper input format for images (file upload, base64 encoding, or URL references)

**Implementation Tips:**

Both programs should demonstrate robust practices:
- **Security**: Follow Authorization header best practices from the [HTTP Request](#http-request) section by using environment variables for API keys instead of hardcoding them
- **Transparency**: Implement using the `requests` package following the request-response patterns shown in the [Interact with APIs with Python](#interact-with-apis-with-python) section, which provides better understanding of HTTP fundamentals than provider-specific SDKs
