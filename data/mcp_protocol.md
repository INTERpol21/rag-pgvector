# MCP: Model Context Protocol

The Model Context Protocol (MCP) is an open standard for connecting AI
applications to external tools and data sources. It plays the role USB
played for peripherals: one connector instead of a custom integration per
assistant-tool pair.

## Architecture

MCP defines three roles. A **host** is the AI application the user faces
(an IDE assistant, a chat app). The host runs one **client** per connection,
and each client talks to exactly one **server** — a process that exposes
capabilities. Messages are JSON-RPC 2.0: requests carry `id`, `method` and
`params`; responses carry `result` or `error`; notifications have no `id`.

## Capabilities

Servers advertise three kinds of capabilities:

- **Tools** — model-controlled actions with a name, description and a JSON
  Schema for arguments (`tools/list`, `tools/call`). The LLM decides when to
  invoke them; results return as content blocks (text, images, resources).
- **Resources** — application-controlled data identified by URIs
  (`resources/list`, `resources/read`), such as files, tickets or table
  schemas the host can attach as context.
- **Prompts** — user-controlled templates (`prompts/list`, `prompts/get`)
  that appear as slash-commands or menu entries in the host UI.

## Lifecycle

A session starts with an `initialize` handshake in which client and server
exchange protocol versions and capability sets, followed by an
`initialized` notification. After that, the client may list and call tools,
subscribe to resource updates, and receive `notifications/tools/list_changed`
when the server's toolset evolves.

## Transports

Two standard transports exist. **stdio** runs the server as a local child
process, framing JSON-RPC messages over stdin/stdout — ideal for local
tools. **Streamable HTTP** exposes a single HTTP endpoint; the server may
upgrade responses to Server-Sent Events to stream results and server-initiated
messages to remote clients. Custom transports are permitted as long as they
preserve JSON-RPC semantics.

## Security considerations

Tools are arbitrary code execution by invitation, so hosts must show users
what a tool will do and require consent before calling it. Servers should
validate arguments against their declared schemas, apply least-privilege
credentials, and treat resource contents as untrusted input. Prompt
injection through tool results — a document that instructs the model to
exfiltrate secrets — is the canonical attack to design against.
