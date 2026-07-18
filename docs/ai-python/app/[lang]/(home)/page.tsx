import { DynamicCodeBlock } from "fumadocs-ui/components/dynamic-codeblock";
import type { Metadata } from "next";
import {
  CommandPromptContent,
  CommandPromptCopy,
  CommandPromptList,
  CommandPromptPrefix,
  CommandPromptRoot,
  CommandPromptSurface,
  CommandPromptTrigger,
  CommandPromptTriggerDivider,
  CommandPromptViewport,
} from "@vercel/geistdocs/components/command-prompt";
import { CenteredSection } from "./components/centered-section";
import { CTA } from "./components/cta";
import { Hero } from "./components/hero";
import { OneTwoSection } from "./components/one-two-section";
import { Templates } from "./components/templates";
import { TextGridSection } from "./components/text-grid-section";

const title = "AI SDK for Python";
const description =
  "Build LLM-powered applications and agent loops";

export const metadata: Metadata = {
  title,
  description,
};

const templates = [
  {
    title: "Template 1",
    description: "Description of template 1",
    link: "https://example.com/template-1",
    image: "https://placehold.co/600x400.png",
  },
  {
    title: "Template 2",
    description: "Description of template 2",
    link: "https://example.com/template-2",
    image: "https://placehold.co/600x400.png",
  },
  {
    title: "Template 3",
    description: "Description of template 3",
    link: "https://example.com/template-3",
    image: "https://placehold.co/600x400.png",
  },
];

const textGridSection = [
  {
    id: "1",
    title: "Lean",
    description: "Composable API that works for simple and complex agents",
  },
  {
    id: "2",
    title: "Async",
    description: "Intuitive concurrency, streaming, and human in the loop",
  },
  {
    id: "3",
    title: "Versatile",
    description: "Deploy as long-running, serverless, or durable",
  },
];

const COMMAND_FOR_HUMANS = "uv add ai";
const COMMAND_FOR_AGENTS = "npx skills add vercel-labs/ai-python";
const DEFAULT_AGENT_LOOP_CODE = `class CustomAgent(ai.Agent):

    async def loop(self, context: ai.Context):
        # Custom event loop implementation for advanced use cases
        while context.keep_running():
            async with (
                ai.stream(context=context) as stream,
                ai.ToolRunner() as tr,
            ):
                # Process the LLM stream and concurrently start running
                # tool calls as they come
                async for event in ai.util.merge(stream, tr.events()):
                    # Append the event to the history
                    yield event

                    if isinstance(event, ai.events.ToolEnd):
                        # Schedule the tool call
                        tr.schedule(
                            context.resolve(event.tool_call)
                        )

                context.add(stream.message)
                context.add(tr.get_tool_message())`;

const STREAM_TO_AGENT_CODE = `# Low-level streaming API
async with ai.stream(model, [ai.user_message("Hello!")]) as s:
    async for event in s:
        print(event)

# High level agent-building API with tool calling and hooks
async with agent.run(model, [ai.user_message("Robot uprising?")]) as s:
    async for event in s:
        print(event)`;

const HomePage = () => (
  <div className="container mx-auto max-w-5xl">
    <Hero
      badge="Now in Beta"
      description={description}
      title={title}
    >
      <CommandPromptRoot defaultValue="humans">
        <CommandPromptList>
          <CommandPromptTrigger className="min-w-[90px]" value="humans">
            For humans
          </CommandPromptTrigger>
          <CommandPromptTriggerDivider />
          <CommandPromptTrigger className="min-w-[84px]" value="agents">
            For agents
          </CommandPromptTrigger>
        </CommandPromptList>
        <CommandPromptSurface>
          <CommandPromptPrefix>$</CommandPromptPrefix>
          <CommandPromptViewport>
            <CommandPromptContent copyValue={COMMAND_FOR_HUMANS} value="humans">
              {COMMAND_FOR_HUMANS}
            </CommandPromptContent>
            <CommandPromptContent copyValue={COMMAND_FOR_AGENTS} value="agents">
              {COMMAND_FOR_AGENTS}
            </CommandPromptContent>
          </CommandPromptViewport>
          <CommandPromptCopy />
        </CommandPromptSurface>
      </CommandPromptRoot>
    </Hero>
    <div className="grid divide-y border-y sm:border-x">
      <TextGridSection data={textGridSection} />
      <CenteredSection
        description="Primitives for streaming, tool dispatch, and loop execution control, joined together using (mostly) plain Python."
        title="An agent loop you can read"
      >
        <DynamicCodeBlock
          code={DEFAULT_AGENT_LOOP_CODE}
          codeblock={{
            "data-line-numbers": true,
            className: "mx-auto my-0 w-full max-w-4xl text-left",
            title: "agent.py",
          }}
          lang="python"
        />
      </CenteredSection>
      <OneTwoSection
        description="Build more AI apps with less framework."
        title="Only essentials"
      >
        <DynamicCodeBlock
          code={STREAM_TO_AGENT_CODE}
          codeblock={{
            "data-line-numbers": true,
            className: "my-0",
            title: "agent.py",
          }}
          lang="python"
        />
      </OneTwoSection>
      {/* <Templates */}
      {/*   data={templates} */}
      {/*   description="See Geistdocs in action with one of our templates." */}
      {/*   title="Get started quickly" */}
      {/* /> */}
      <CTA cta="Get started" href="/docs" title="Build your agent today" />
    </div>
  </div>
);

export default HomePage;
