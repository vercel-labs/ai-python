import type { GeistdocsAgentReadinessConfig } from "@vercel/geistdocs/config";

export const Logo = () => (
  <span className="font-semibold text-gray-1000 text-lg leading-none tracking-[-3%]">
    AI SDK for Python
  </span>
);

export const github = {
  branch: "main",
  editPath: "docs/ai-python/content/docs/{path}",
  owner: "vercel-labs",
  repo: "ai-python",
};

export const nav = [
  {
    label: "Docs",
    href: "/docs",
  },
  {
    label: "Source",
    href: `https://github.com/${github.owner}/${github.repo}/`,
  },
];

export const suggestions = [
  "How do I stream a model response?",
  "How do I define tools for an agent?",
  "How do I build a custom agent loop?",
  "How do I connect an agent to AI SDK UI?",
];

export const title = "AI SDK for Python Documentation";

export const prompt =
  "You are a helpful assistant specializing in answering questions about the AI SDK for Python, toolkit for building LLM-powered applications and agent loops.";

export const agent = {
  product: {
    name: "AI SDK for Python",
    description:
      "The AI SDK for Python is a toolkit for building LLM-powered applications and agent loops with composable primitives: models, messages, streams, tools, agents, and hooks.",
    category: "SDK",
    audience: ["Python developers", "AI application engineers"],
    useCases: [
      "Build agent loops with tools and hooks",
      "Stream LLM responses across providers",
      "Connect Python agents to AI SDK UI frontends",
    ],
  },
  links: [
    {
      label: "AI SDK for Python source",
      href: `https://github.com/${github.owner}/${github.repo}`,
      description: "Source repository for the AI SDK for Python",
    },
  ],
} satisfies GeistdocsAgentReadinessConfig;

export const translations = {
  en: {
    displayName: "English",
  },
};

export const basePath: string | undefined = undefined;

/**
 * Unique identifier for this site, used in markdown request tracking analytics.
 * Each site using geistdocs should set this to a unique value (e.g. "ai-sdk-docs", "next-docs").
 */
export const siteId: string | undefined = undefined;
