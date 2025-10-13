from langchain_core.prompts import PromptTemplate

PLANNER_PROMPT_TEMPLATE = PromptTemplate(
    template="""Given a user query, create a plan to solve it with the utmost parallelism.

The plan should comprise a sequence of actions from the following {tool_count} types:
{tool_descriptions}

USER QUERY: {user_query}

IMPORTANT: Use exact tool names: {tool_names}

GUIDELINES:
- Each action described above contains input/output types and description.
  - You must strictly adhere to the input and output types for each action.
  - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.
- Each action in the plan should strictly be one of the above types. Follow the conventions for each action.
- Each action MUST have a unique ID, which is strictly increasing.
- Inputs for actions can either be constants or outputs from preceding actions. In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.
- Ensure the plan maximizes parallelism.
- Only use the provided action types. If a query cannot be addressed using these, explain what additional tools would be needed.
- Never introduce new actions other than the ones provided.

DEPENDENCIES - CRITICAL:
For EACH task, ask two questions:

1. Does this task use output ($N) from earlier tasks?
   → Add those task numbers to deps

2. Does this task need anything created/established by earlier tasks?
   → Look at EACH parameter value - does it depend on something an earlier task creates?
   → Add those task numbers to deps

Include ALL dependencies in (deps: [...]) - both data and ordering.
If a task needs nothing from earlier tasks, use (deps: [])

FORMAT: N. tool_name(param='value', other='$N') (deps: [all, dependency, numbers])""",
    input_variables=[
        "tool_count",
        "tool_descriptions",
        "user_query",
        "tool_names",
    ],
)
