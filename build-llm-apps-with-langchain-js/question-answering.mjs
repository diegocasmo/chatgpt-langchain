import "dotenv/config";

import * as parser from "pdf-parse/lib/pdf-parse.js";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

/*
 * Use an in memory vectorstore and retrieve info from it
 */
const pdfLoader = new PDFLoader(
  "./vasco-developing-ai-crawlers-for-ml-blink.pdf"
);

const rawCS229Docs = await pdfLoader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1536,
  chunkOverlap: 128,
});

const splitDocs = await splitter.splitDocuments(rawCS229Docs);

// Add the documents to the vectorstore
const embeddings = new OpenAIEmbeddings();
const vectorstore = new MemoryVectorStore(embeddings);
await vectorstore.addDocuments(splitDocs);

/*
 * Document retrieval chain
 */
const retriever = vectorstore.asRetriever();
const transformDocsToString = (documents) =>
  documents.map((doc) => `<doc>\n${doc.pageContent}\n</doc>`).join("\n");

const documentRetrievalChain = RunnableSequence.from([
  (input) => input.question,
  retriever,
  transformDocsToString,
]);

/*
 * Augmented generation
 */
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-1106",
});

const TEMPLATE_STRING = `You are an experienced researcher,
expert at interpreting and answering questions based on provided sources.
Using the provided context, answer the user's question
to the best of your ability using only the resources provided.
Be verbose!

<context>

{context}

</context>

Now, answer this question using the above context:

{question}`;

const answerGenerationPrompt = ChatPromptTemplate.fromTemplate(TEMPLATE_STRING);

const retrievalChain = RunnableSequence.from([
  {
    context: documentRetrievalChain,
    question: (input) => input.question,
  },
  answerGenerationPrompt,
  model,
  new StringOutputParser(),
]);

const answer = await retrievalChain.invoke({
  question: "What is ML-Blink?",
});

console.log(answer);
