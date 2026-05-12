import { defineConfig } from "@vercel/geistdocs/config";
import {
  basePath,
  github,
  Logo,
  nav,
  prompt,
  siteId,
  suggestions,
  title,
  translations,
} from "@/geistdocs";

export const config = defineConfig({
  title,
  defaultLanguage: "en",
  logo: <Logo />,
  github,
  nav,
  basePath,
  siteId,
  translations,
  ai: {
    prompt,
    suggestions,
  },
});
