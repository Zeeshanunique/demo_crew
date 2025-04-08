'use client';

import { useEffect, useState } from 'react';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Dataset } from '@/lib/dataset-utils';
import { AlertCircle, Check, Loader2 } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

interface DatasetManipulatorProps {
  dataset: Dataset;
  onDatasetUpdated?: (newDataset: Dataset) => void;
}

export default function DatasetManipulator({ dataset, onDatasetUpdated }: DatasetManipulatorProps) {
  const [apiKey, setApiKey] = useState<string>('');
  const [prompt, setPrompt] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<boolean>(false);

  useEffect(() => {
    // Try to load API key from localStorage
    try {
      const savedApiKey = localStorage.getItem('gemini_api_key');
      if (savedApiKey) {
        setApiKey(savedApiKey);
      }
    } catch (e) {
      console.error('Failed to load API key from localStorage');
    }
  }, []);

  const handleSaveApiKey = () => {
    try {
      localStorage.setItem('gemini_api_key', apiKey);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (e) {
      console.error('Failed to save API key');
    }
  };

  const handleManipulateDataset = async () => {
    if (!apiKey || !prompt) {
      setError('Please provide both an API key and a manipulation prompt');
      return;
    }

    setLoading(true);
    setError(null);
    setResult('');

    try {
      const genAI = new GoogleGenerativeAI(apiKey);
      const model = genAI.getGenerativeModel({ model: "gemini-pro" });

      // Create a prompt that includes the dataset and the user's instructions
      const datasetJson = JSON.stringify(dataset, null, 2);
      const fullPrompt = `
        You are a data manipulation assistant. I have a dataset in JSON format that I want you to manipulate.
        
        Here is the dataset:
        \`\`\`json
        ${datasetJson}
        \`\`\`
        
        Instructions for manipulation:
        ${prompt}
        
        Please provide the resulting JSON after manipulation. Only return valid JSON that follows the same structure as the input (with "results" array containing objects with "output" and "agent_type" fields), nothing else.
      `;

      const result = await model.generateContent(fullPrompt);
      const response = result.response;
      const text = response.text();
      
      let processedText = text;
      
      // Extract JSON from the response if it's wrapped in markdown code blocks
      const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
      if (jsonMatch && jsonMatch[1]) {
        processedText = jsonMatch[1];
      }
      
      try {
        const newDataset = JSON.parse(processedText);
        if (onDatasetUpdated && typeof onDatasetUpdated === 'function') {
          onDatasetUpdated(newDataset);
        }
        setResult(JSON.stringify(newDataset, null, 2));
      } catch (jsonError) {
        console.error('Failed to parse AI response as JSON:', jsonError);
        setError('The AI did not return valid JSON. Please try again with a different prompt.');
        setResult(processedText);
      }
    } catch (e: any) {
      console.error('Error during dataset manipulation:', e);
      setError(`Manipulation failed: ${e.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Dataset Manipulator</CardTitle>
        <CardDescription>
          Use AI to transform, filter, or analyze your dataset
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Google AI API Key</label>
          <div className="flex space-x-2">
            <Input
              type="password"
              placeholder="Enter your Google AI API Key"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="flex-1"
            />
            <Button onClick={handleSaveApiKey} disabled={!apiKey}>
              {success ? <Check className="h-4 w-4" /> : 'Save'}
            </Button>
          </div>
          <p className="text-sm text-gray-500 mt-1">
            Required for dataset manipulation. Get a key at{' '}
            <a 
              href="https://ai.google.dev/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-blue-500 hover:underline"
            >
              ai.google.dev
            </a>
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Manipulation Instructions</label>
          <Textarea
            placeholder="Enter instructions for how to manipulate the dataset (e.g., 'Filter results to only include image data', 'Summarize each output to 50 words or less')"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
          />
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {result && (
          <div>
            <label className="block text-sm font-medium mb-1">Result</label>
            <div className="bg-gray-900 p-4 rounded-md overflow-auto max-h-[400px]">
              <pre className="text-gray-300 text-sm whitespace-pre-wrap">{result}</pre>
            </div>
          </div>
        )}
      </CardContent>
      <CardFooter>
        <Button 
          onClick={handleManipulateDataset} 
          disabled={loading || !apiKey || !prompt}
          className="w-full"
        >
          {loading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Processing...
            </>
          ) : (
            'Manipulate Dataset'
          )}
        </Button>
      </CardFooter>
    </Card>
  );
}
