import "dotenv/config";

import * as parser from "pdf-parse/lib/pdf-parse.js";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { similarity } from "ml-distance";

/*
 * Vectorstore ingestion
 */
const embeddings = new OpenAIEmbeddings();
await embeddings.embedQuery("This is some sample text");

const vector1 = await embeddings.embedQuery(
  "What are vectors useful for in machine learning?"
);
const unrelatedVector = await embeddings.embedQuery(
  "A group of parrots is called a pandemonium."
);
console.log(similarity.cosine(vector1, unrelatedVector));

const similarVector = await embeddings.embedQuery(
  "Vectors are representations of information."
);

console.log(similarity.cosine(vector1, similarVector));

/*
 * Use an in memory vectorstore and retrieve info from it
 */
const pdfLoader = new PDFLoader(
  "./vasco-developing-ai-crawlers-for-ml-blink.pdf"
);

const rawCS229Docs = await pdfLoader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 128,
  chunkOverlap: 0,
});

const splitDocs = await splitter.splitDocuments(rawCS229Docs);

// Add the documents to the vectorstore
const vectorstore = new MemoryVectorStore(embeddings);
await vectorstore.addDocuments(splitDocs);

// Retrieve similar documents
const retrievedDocs = await vectorstore.similaritySearch(
  "What is ML-Blink?",
  4
);

const pageContents = retrievedDocs.map((doc) => doc.pageContent);

console.log(pageContents);

/*
 * Retrievers
 */
const retriever = vectorstore.asRetriever();
console.log(await retriever.invoke("What is ML-Blink?"));
