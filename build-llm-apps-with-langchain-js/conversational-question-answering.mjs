import "dotenv/config";

import * as parser from "pdf-parse/lib/pdf-parse.js";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { RunnablePassthrough } from "@langchain/core/runnables";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";

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
  temperature: 0.1,
});

const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `Given the following conversation and a follow up question,
rephrase the follow up question to be a standalone question.`;

const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human",
    "Rephrase the following question as a standalone question:\n{question}",
  ],
]);

const rephraseQuestionChain = RunnableSequence.from([
  rephraseQuestionChainPrompt,
  model,
  new StringOutputParser(),
]);

const originalQuestion = "What further research does this paper recommend?";

const ANSWER_CHAIN_SYSTEM_TEMPLATE = `You are an experienced researcher,
expert at interpreting and answering questions based on provided sources.
Using the below provided context and chat history,
answer the user's question to the best of
your ability
using only the resources provided. Be verbose!

<context>
{context}
</context>`;

const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human",
    "Now, answer this question using the previous context and chat history:\n{standalone_question}",
  ],
]);

const conversationalRetrievalChain = RunnableSequence.from([
  RunnablePassthrough.assign({
    standalone_question: rephraseQuestionChain,
  }),
  RunnablePassthrough.assign({
    context: documentRetrievalChain,
  }),
  answerGenerationChainPrompt,
  model,
  new StringOutputParser(),
]);

const messageHistory = new ChatMessageHistory();

const retrievalChain = new RunnableWithMessageHistory({
  runnable: conversationalRetrievalChain,
  getMessageHistory: (_sessionId) => messageHistory,
  historyMessagesKey: "history",
  inputMessagesKey: "question",
});

await retrievalChain.invoke(
  {
    question: originalQuestion,
  },
  {
    configurable: { sessionId: "test" },
  }
);

const finalResult = await retrievalChain.invoke(
  {
    question: "Can you list them in bullet point form?",
  },
  {
    configurable: { sessionId: "test" },
  }
);

console.log(finalResult);
