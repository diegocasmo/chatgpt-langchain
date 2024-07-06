import "dotenv/config";

import * as parser from "pdf-parse/lib/pdf-parse.js";
import ignore from "ignore";
import { GithubRepoLoader } from "@langchain/community/document_loaders/web/github";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

/*
 * Github loader
 */
const gitHubLoader = new GithubRepoLoader(
  "https://github.com/langchain-ai/langchainjs",
  {
    recursive: false,
    ignorePaths: ["*.md", "yarn.lock"],
  }
);

const docs = await gitHubLoader.load();
console.log(docs.slice(0, 3));

/*
 * PDF loader
 */
const pdfLoader = new PDFLoader(
  "./vasco-developing-ai-crawlers-for-ml-blink.pdf"
);

const rawCS229Docs = await pdfLoader.load();
console.log(rawCS229Docs.slice(0, 5));

/*
 * Splitting
 */
const splitter = RecursiveCharacterTextSplitter.fromLanguage("js", {
  chunkSize: 512,
  chunkOverlap: 54,
});

const splitDocs = await splitter.splitDocuments(rawCS229Docs);
console.log(splitDocs.slice(0, 5));
