# Development & Architecture Rules

## 1. Modularity & Clean Code
- **Single Responsibility:** Every function and module must do one thing well. Keep functions short (under 30 lines where possible).
- **Separation of Concerns:** Strictly separate business logic from I/O, database calls, UI components, and external API integrations.
- **No Monoliths:** Break complex tasks into small, reusable helper functions or separate files before writing the main implementation.
- **Explicit Interfaces:** Define clear input/output types (e.g., TypeScript interfaces, Python type hints) for all core functions before implementing them.
- **DRY (Don't Repeat Yourself):** If logic is repeated more than twice, abstract it into a shared utility.

## 2. Testing Workflow (Mandatory)
- **Test-After-Core:** Immediately after implementing any core function or business logic module, you MUST write unit tests for it before moving to the next task.
- **Edge Cases First:** Every test suite must cover:
  - Happy path (standard expected input).
  - Edge cases (empty inputs, nulls, boundary values).
  - Error handling (invalid inputs, failed dependencies).
- **Run & Verify:** Always run the test suite via CLI after writing tests. Do not consider a task complete until all tests pass.
- **Mock External Dependencies:** Isolate unit tests by mocking databases, network requests, and file system operations.

## 3. Execution & Planning Protocol
- **Plan Before Coding:** For any task involving more than 2 files, outline your modular design and file structure first, and wait for my approval.
- **Incremental Steps:** Build and verify one module at a time. Never generate massive, untested multi-file changes in a single step.
- **Self-Review:** Before finishing a task, review your own diff to ensure no dead code, unused imports, or hardcoded secrets remain.
