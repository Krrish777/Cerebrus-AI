## Operational Rules (Hard Requirements)

### Do
1. **All tests** must be written inside `tests/` using a clear, modular structure.  
2. **Always activate the virtual environment** before executing or generating any runnable code (`source venv/bin/activate` or Windows equivalent).  
3. **Use logging consistently**. Every Python file must import and use `customlogger` with meaningful log messages.  
4. **Use `data/` as the sole source of test data**.
5. **Always use uv as a package manager** 
5. **Follow all design principles strictly** (see below).  

### Don’t
1. Never create or modify files outside the designated directories (`src/`, `tests/`, `data/`, etc.).  
2. Never include emojis, decorative symbols, or non-standard characters in logs.  
3. Never generate runnable code without ensuring the virtual environment is active.  
4. Never hard-code values such as file paths, credentials, constants, or environment-specific settings. Always use configs, environment variables, or parameters.  

---

## Design Principles (Enforced)

### 1. Cohesion & Single Responsibility
- A class or module must have one purpose.  
- If its reason to change multiplies, split it.  
**Copilot must check:**  
“If I need to add a new behavior, would this class explode in complexity?” If yes, restructure.

### 2. Encapsulation & Abstraction  
- Internal state must stay private.  
- Only expose clear methods/components.  
- Replace direct data access with well-defined interfaces.  
**Copilot must ensure:**  
Changing internals never breaks external callers.

### 3. Loose Coupling & Modularity  
- Use dependency injection.  
- Never instantiate dependencies deep inside classes.  
- Favor interfaces/ABCs for swappable components.  
**Copilot must ask:**  
“Can this component be tested in isolation without bootstrapping the whole system?”

### 4. Extensibility Without Modification  
- New features should plug in smoothly without editing core logic.  
- Favor composition, strategy pattern, and plug-in modules.  
- No monolithic if/else trees for behavior.  
**Copilot must guarantee:**  
Feature additions look like new modules, not edits to old ones.

### 5. Portability  
- Use `pathlib` for all paths.  
- All config must come from environment variables or config files.  
- No OS-specific code.  
**Copilot must check:**  
“Will this run unchanged on Linux, macOS, and Windows?”

### 6. Defensibility  
- Validate all inputs immediately.  
- Fail fast with explicit errors.  
- Log only safe, non-sensitive details.  
- Use conservative defaults for all settings.  
**Copilot must verify:**  
“What is the worst-case failure if someone passes garbage here?”

### 7. Maintainability & Testability  
- Code must be simple, readable, and deterministic where possible.  
- Separate I/O from business logic.  
- Pure functions whenever possible.  
- Every important path must have test coverage.  
**Copilot should enforce:**  
If writing tests is painful, the design is wrong.

### 8. Simplicity (KISS, DRY, YAGNI)  
- Prefer simple over clever.  
- No duplicate logic.  
- No speculative abstractions.  
**Copilot must reject:**  
Any feature that isn’t required now.

---

## Forbidden Code Smells  
Copilot must avoid generating code that has:
- God classes/multi-purpose modules  
- Public mutable fields  
- Hardcoded dependencies  
- Hardcoded paths  
- Deep nested conditionals  
- Silent failure or swallowed exceptions  
- Giant functions (>50 lines)  
- Repeated blocks of logic  
- “Future-proofing” abstractions that serve no current purpose  

---

## File & Directory Rules
- `src/` contains implementation  
- `tests/` contains all tests  
- `data/` contains test fixtures only  
- `config/` holds configuration files  
- No random file creation  
- No writing outside project root  

---

## Style & Syntax Standards  
(Specifically for Python coding.)


### 2.1 Imports  
- One import per line.  
- Use absolute imports unless very compelling reason.  
- No `from module import *`.  
- Group imports: standard library, third-party, local modules. Blank line between groups.

### 2.2 Naming Conventions  
- Classes: `CamelCase`.  
- Functions/variables: `lower_case_with_underscores`.  
- Constants: `ALL_CAPS_WITH_UNDERSCORES`.  
- Modules: all-lowercase, underscores if needed.  
- Private attributes/methods: single leading underscore `_`.  
- Avoid ambiguous names like `data`, `info`, `utils`; prefer descriptive.

### 2.3 Type Hints  
- Use type hints on all public functions/methods.  
- Use `typing` for complex types (`Optional`, `Union`, `Tuple`, `Iterable`).  
- If function returns nothing, mark `-> None`.

---

## Logging Standards

- Use standard log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
* No emojis or decorative chars in log messages.
* Do not log sensitive data.
* Good example:

  ```python
  logger.info("Processing %d records from %s", len(records), source_path)
  ```
* Bad example:

  ```python
  logger.info("We’re done 🎉")  # ❌  
  logger.debug(f"User password: {password}")  # ❌  
  ```

---

## Testing Standards
* All test modules must reside under tests/, named according to test_*.py.

* Test input data (fixtures) should live in data/ and be accessed via code, e.g.:
```python
from pathlib import Path
import pytest

@pytest.fixture
def sample_input():
    return (Path("data") / "sample_input.json").read_text(encoding="utf-8")
```

- Tests must be fast, isolated, and deterministic. No real external I/O, network calls or DB calls unless explicitly mocked.

- Use pytest fixtures + parametrization and mocking properly:

    - Use @pytest.fixture, scoped appropriately (function, module, session).

    - Use @pytest.mark.parametrize to test multiple inputs/outputs instead of copying tests. 

    - Use mocking frameworks (e.g., pytest-mocker or unittest.mock) to stub external dependencies and isolate logic. 

- Test names must clearly reflect behavior under test, e.g.
```python
def test_user_repository_returns_empty_when_no_user():
    ...
```

- Avoid duplication in tests: extract common set-up via fixtures or helper functions (DRY principle). Use fixtures or parametrize rather than copy-pasting logic. 

- Coverage threshold: critical modules must achieve ≥ 90% branch coverage (or your defined standard) before merge.

---

## Configuration & Environment

* No hard-coded configuration values. Use environment variables (`os.environ`) or files in `config/`.
* Use `pathlib.Path` for all filesystem references:

  ```python
  from pathlib import Path
  data_dir = Path(__file__).parent.parent / "data"
  ```

  Avoid `"data/" + "file.txt"`.
* No OS‐specific assumptions (no `C:/`, no backslashes in paths).

---

## Example Code – Good vs Bad

### Good Example

```python
# src/services/processor.py
from pathlib import Path
from typing import List, Dict
from src.logging.customlogger import get_logger

logger = get_logger(__name__)

class RecordProcessor:
    """
    Processes records from provided file paths.
    """

    def __init__(self, repository: "RecordRepository"):
        self._repository = repository

    def load_records(self, file_path: Path) -> List[Dict[str, str]]:
        """
        Load records from given JSON file.
        :param file_path: Path to JSON file
        :return: List of records as dicts
        :raises FileNotFoundError: if file does not exist
        """
        if not file_path.exists():
            logger.error("Record file not found: %s", file_path)
            raise FileNotFoundError(f"Missing record file: {file_path}")

        logger.debug("Loading records from %s", file_path)
        content = file_path.read_text(encoding="utf-8")
        # parse JSON content (omitted)
        records = []  # placeholder
        return records

    def process(self, records: List[Dict[str, str]]) -> int:
        """
        Process the list of records and store results.
        :param records: list of record dicts
        :return: number of processed records
        """
        logger.info("Processing %d records", len(records))
        count = 0
        for record in records:
            # business logic here
            self._repository.save(record)
            count += 1
        logger.info("Processed %d records", count)
        return count
```

### Bad Example

```python
# src/services/processor.py
import json

class Processor:
    def __init__(self, repo):
        self.repo = repo

    def run(self, path):
        f = open("data/file.json")  # hardcoded path
        data = json.load(f)
        f.close()
        for d in data:
            self.repo.store(d)  # unclear what store does
        print("done")  # prints instead of logging
```

Issues: hard-coded path, no logging infrastructure, mixing I/O and business logic, poor naming, no type hints.

---

## Meta Rules for Agent Behavior

1. Always justify design decisions according to the principles above.
2. If a user request violates the principles, the agent must issue a warning *and* propose an alternative that aligns with standards.
3. When expanding or generating code, adhere strictly to directory structure and naming conventions.
4. No placeholder logic without a clearly documented TODO and reference to a task or issue.

---

## Quick Decision Tree

```
Is my class doing more than one thing?
├─ Yes → Split it (SRP)
└─ No → ✓

Can clients modify my internal state?
├─ Yes → Make it private (Encapsulation)
└─ No → ✓

Am I creating dependencies inside my class?
├─ Yes → Inject them (Loose Coupling)
└─ No → ✓

Do I need to edit existing code to add features?
├─ Yes → Use strategy/plugin pattern (Extensibility)
└─ No → ✓

Do I have hard-coded paths or platform assumptions?
├─ Yes → Use pathlib and config (Portability)
└─ No → ✓

Am I accepting input without validation?
├─ Yes → Validate and fail-fast (Defensibility)
└─ No → ✓

Would this be hard to test?
├─ Yes → Separate logic from I/O (Testability)
└─ No → ✓

Am I adding abstractions “just in case”?
├─ Yes → Remove it (YAGNI/KISS)
└─ No → ✓

Am I repeating logic?
├─ Yes → Extract it (DRY)
└─ No → ✓
```
