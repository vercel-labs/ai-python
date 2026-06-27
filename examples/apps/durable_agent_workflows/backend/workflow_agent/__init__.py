import vercel.workflow

# The app uses one registry for all workflow decorators so queue messages
# are dispatched by the same Workflows instance.
workflow = vercel.workflow.Workflows()
