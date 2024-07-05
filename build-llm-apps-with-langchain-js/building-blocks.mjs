import "dotenv/config";

import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-1106",
  apiKey: process.env.OPENAI_API_KEY,
});

const result = await model.invoke([new HumanMessage("Tell me a joke.")]);
console.log(result);
