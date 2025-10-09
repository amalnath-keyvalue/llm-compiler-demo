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

DEPENDENCIES: Use $N to reference previous task outputs.
Example: tool_name(param='$2') uses output from task 2.

PLANNING: Break tasks into logical steps with dependencies:
- When one task produces output that another task needs as input, use $N to reference it
- Create dependencies to form an efficient workflow
- Independent tasks can run in parallel

CRITICAL: Always use dependencies when one task's output is needed by another!
- If task A produces output, and task B needs that output, use $A in task B
- Generate content for EACH file separately - don't generate everything at once
- Create dependencies to form an efficient workflow
- This creates a DAG where tasks execute based on dependencies, not plan order

Format: N. tool_name(param='value', other='$N') (deps: [1, 2, 3])""",
    input_variables=[
        "tool_count",
        "tool_descriptions",
        "user_query",
        "tool_names",
    ],
)

JOIN_PROMPT_TEMPLATE = PromptTemplate(
    template="""Synthesize task execution results into a coherent response.

Original user query: {user_query}

Task execution results:
{results_text}

Please provide a comprehensive, well-structured response that addresses the user's original query based on these task results. 
Be specific about what was accomplished and provide any relevant details from the task outputs.""",
    input_variables=["user_query", "results_text"],
)

SHOULD_CONTINUE_PROMPT_TEMPLATE = PromptTemplate(
    template="""Determine whether a task execution is complete or needs re-planning.

Original user query: {user_query}

Current response: {latest_response}

Based on the original query and the current response, decide if:
1. The task is COMPLETE and satisfactory (return "END")
2. The task needs RE-PLANNING because something is missing or incorrect (return "REPLAN")

Consider:
- Does the response fully address the user's query?
- Are there any obvious gaps or issues?
- Would additional tasks improve the result?

Respond with only "END" or "REPLAN".""",
    input_variables=[
        "user_query",
        "latest_response",
    ],
)

REPLANNER_PROMPT_TEMPLATE = PromptTemplate(
    template="""The previous execution was insufficient to fully address the user's query. 
Create a new plan that builds upon the current results.

The plan should comprise a sequence of actions from the following {tool_count} types:
{tool_descriptions}

USER QUERY: {user_query}

PREVIOUS EXECUTION RESULTS:
{results_text}

CURRENT RESPONSE: {latest_response}

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
- Build upon the existing results and address any gaps or issues identified.

DEPENDENCIES: Use $N to reference previous task outputs.
Example: tool_name(param='$2') uses output from task 2.

REPLANNING: Create additional tasks that:
- Address gaps in the previous execution
- Use different approaches if needed
- Build upon existing results
- Maximize parallelism

Format: N. tool_name(param='value', other='$N') (deps: [1, 2, 3])""",
    input_variables=[
        "tool_count",
        "tool_descriptions",
        "user_query",
        "results_text",
        "latest_response",
        "tool_names",
    ],
)
