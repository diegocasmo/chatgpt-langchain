import "dotenv/config";

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { HumanMessage } from "@langchain/core/messages";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-1106",
  apiKey: process.env.OPENAI_API_KEY,
});

/*
 * Using human messages directly
 */
// console.log(await model.invoke([new HumanMessage("Tell me a joke.")]));

/*
 * Using prompt templates
 */
// const prompt = ChatPromptTemplate.fromTemplate(
//   `What are three good names for a company that makes {product}?`
// );

// console.log(
//   await model.invoke(
//     await prompt.formatMessages({ product: "colorful socks" })
//   )
// );

/*
 * LangChain Expression Language (LCEL)
 */
// const prompts = ChatPromptTemplate.fromMessages([
//   ["system", "You are an expert at picking company names"],
//   ["human", "What are three good names for a company that makes {product}?"],
// ]);

// const chain = prompts.pipe(model);
// console.log(
//   await chain.invoke({ product: "colorful socks" })
// );

/*
 * Output parses
 */
// const prompts = ChatPromptTemplate.fromMessages([
//   ["system", "You are an expert at picking company names"],
//   ["human", "What are three good names for a company that makes {product}?"],
// ]);
// const outputParser = new StringOutputParser();
// const chain = RunnableSequence.from([prompts, model, outputParser]);
// console.log(await chain.invoke({ product: "fancy cookies" }));
