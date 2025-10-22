import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AzureOpenAI

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    citations: list = Field(default_factory=list)

def get_azure_openai_response(text: str, show_citations: bool = False):
    load_dotenv()
    azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
    azure_oai_key = os.getenv("AZURE_OAI_KEY")
    azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
    azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    azure_search_key = os.getenv("AZURE_SEARCH_KEY")
    azure_search_index = os.getenv("AZURE_SEARCH_INDEX")

    client = AzureOpenAI(
        azure_endpoint=azure_oai_endpoint,
        api_key=azure_oai_key,
        api_version="2025-01-01-preview"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Finley, an AI-powered financial advisor assistant. "
                "Always refer to yourself as 'Finley' in every response. "
                "You are knowledgeable, concise, and professional — capable of explaining financial policies, interpreting stock reports, retrieving company filings, and analyzing both structured and unstructured financial data. "
                "You provide clear, actionable, and unbiased insights without offering personal opinions or investment advice. "
                "Your tone is intelligent, confident, and respectful — never casual, overly technical, or speculative. "
                "Always prioritize accuracy, compliance, and clarity. "
                "When unsure or when asked for regulated financial advice, clearly state your limitations and recommend consulting a licensed human advisor. "
            )
        },
        {
            "role": "user",
            "content": text
        }
    ]

    extension_config = {
        "data_sources": [
            {
                "type": "azure_search",
                "parameters": {
                    "endpoint": azure_search_endpoint,
                    "index_name": azure_search_index,
                    "authentication": {
                        "type": "api_key",
                        "key": azure_search_key
                    }
                }
            }
        ]
    }

    response = client.chat.completions.create(
        model=azure_oai_deployment,
        messages=messages,
        max_tokens=1000,
        temperature=0.5,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
        extra_body=extension_config
    )
    # Log the full response for debugging
    print("Full Azure OpenAI response:", response)
    answer = response.choices[0].message.content
    citations = []
    if show_citations:
        # Try to extract citations from context or tool_calls if available
        try:
            # Check for citations in context
            context = getattr(response.choices[0].message, "context", None)
            if context:
                # Sometimes citations are in context["citations"]
                if "citations" in context:
                    citations = context["citations"]
                # Sometimes in context["messages"]
                elif "messages" in context:
                    for msg in context["messages"]:
                        if "citations" in msg:
                            citations.extend(msg["citations"])
                        elif "content" in msg:
                            try:
                                citation_json = json.loads(msg["content"])
                                if "citations" in citation_json:
                                    citations.extend(citation_json["citations"])
                            except Exception:
                                pass
            # Check for citations in tool_calls (if present)
            tool_calls = getattr(response.choices[0].message, "tool_calls", None)
            if tool_calls:
                for call in tool_calls:
                    if "citations" in call:
                        citations.extend(call["citations"])
        except Exception as e:
            print("Citation extraction error:", e)
            citations = []
    return answer, citations

@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    try:
        answer, citations = get_azure_openai_response(request.question, show_citations=True)
        return AskResponse(answer=answer, citations=citations)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

