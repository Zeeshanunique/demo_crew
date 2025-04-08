"use client";

import { useState, useRef, useEffect } from 'react';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Dataset } from "@/lib/dataset-utils";

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface DatasetChatProps {
  dataset: Dataset;
  apiKey?: string;
}

export default function DatasetChat({ dataset, apiKey }: DatasetChatProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Hello! I can help you analyze and query the dataset. What would you like to know about it?'
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [apiKeyInput, setApiKeyInput] = useState(apiKey || '');
  const [keySet, setKeySet] = useState(!!apiKey);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Save API key
  const saveApiKey = () => {
    if (apiKeyInput.trim()) {
      setKeySet(true);
      // Avoid storing API keys in client-side localStorage in a real app
      // This is just for the demo
      try {
        localStorage.setItem('gemini_api_key', apiKeyInput);
      } catch (e) {
        console.error('Failed to store API key');
      }
    }
  };

  const handleSubmit = async () => {
    if (!input.trim() || loading || !keySet) return;
    
    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    setInput('');
    setError(null);

    try {
      // Initialize the Google Generative AI with the API key
      const genAI = new GoogleGenerativeAI(apiKeyInput);
      
      // Create the prompt for the AI
      const datasetJson = JSON.stringify(dataset, null, 2);
      const prompt = `
        You are a helpful assistant that can analyze and provide insights about a dataset.
        Here is the current dataset in JSON format:
        \`\`\`json
        ${datasetJson}
        \`\`\`

        The user's question is: "${input}"
        
        Answer their question based on the dataset. If they ask for modifications to the dataset,
        explain how it could be modified but note that you cannot directly modify the dataset.
      `;
      
      // Get a response from Gemini
      const model = genAI.getGenerativeModel({ model: "gemini-pro" });
      const result = await model.generateContent(prompt);
      const response = result.response.text();
      
      // Add the response to the messages
      const assistantMessage: Message = { role: 'assistant', content: response };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Error querying Generative AI:', err);
      setError('Failed to get a response from the AI. Please check your API key and try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <Card className="w-full flex flex-col h-[600px]">
      <CardHeader>
        <CardTitle>Dataset Chat Assistant</CardTitle>
        <CardDescription>
          Ask questions about the dataset or request analysis
        </CardDescription>
      </CardHeader>
      
      {!keySet ? (
        <CardContent className="flex-grow overflow-hidden flex flex-col">
          <div className="text-center flex-grow flex flex-col items-center justify-center space-y-4">
            <p className="text-gray-400">
              To use the chat assistant, you need to enter your Google Generative AI (Gemini) API key.
            </p>
            <div className="flex w-full max-w-md space-x-2">
              <input
                type="password"
                value={apiKeyInput}
                onChange={(e) => setApiKeyInput(e.target.value)}
                placeholder="Enter your Gemini API key"
                className="flex-grow rounded px-4 py-2 bg-gray-800 border border-gray-700 text-white"
              />
              <Button onClick={saveApiKey}>Save</Button>
            </div>
            <p className="text-sm text-gray-500">
              Your API key is stored locally in your browser and is only used for this application.
            </p>
          </div>
        </CardContent>
      ) : (
        <>
          <CardContent className="flex-grow overflow-hidden">
            <div className="h-full overflow-y-auto px-1 py-2 space-y-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${
                    message.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg p-3 ${
                      message.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-800 text-gray-200'
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="max-w-[80%] rounded-lg p-3 bg-gray-800 text-gray-200">
                    <div className="flex space-x-2">
                      <div className="h-2 w-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="h-2 w-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '200ms' }}></div>
                      <div className="h-2 w-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '400ms' }}></div>
                    </div>
                  </div>
                </div>
              )}
              {error && (
                <div className="bg-red-900/30 text-red-200 p-3 rounded-lg text-sm">
                  {error}
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </CardContent>
          <CardFooter className="border-t border-gray-800 p-3">
            <div className="flex w-full space-x-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about the dataset..."
                className="flex-grow rounded px-3 py-2 bg-gray-800 border border-gray-700 text-white resize-none"
                rows={1}
                disabled={loading}
              />
              <Button 
                onClick={handleSubmit} 
                disabled={loading || !input.trim()}
                className="self-end"
              >
                {loading ? 'Sending...' : 'Send'}
              </Button>
            </div>
          </CardFooter>
        </>
      )}
    </Card>
  );
}