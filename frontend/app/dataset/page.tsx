"use client";

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../components/ui/card";

interface DatasetItem {
  output: string;
  agent_type: string;
}

export default function DatasetPage() {
  const [dataset, setDataset] = useState<DatasetItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDataset = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:8000/api/dataset');
        
        if (!response.ok) {
          throw new Error(`Failed to fetch dataset: ${response.status}`);
        }
        
        const data = await response.json();
        setDataset(data.results || []);
      } catch (err) {
        console.error('Error fetching dataset:', err);
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchDataset();
  }, []);

  // Format the dataset as a JSON string for display
  const formattedDataset = JSON.stringify({ results: dataset }, null, 2);

  return (
    <div className="container mx-auto py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Processed Data Results</h1>
        <p className="text-gray-600">
          Showing simplified dataset with only output and agent_type fields
        </p>
      </div>

      {loading ? (
        <div className="text-center py-12">
          <div className="spinner"></div>
          <p className="mt-4">Loading dataset...</p>
        </div>
      ) : error ? (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
          <p className="font-bold">Error</p>
          <p>{error}</p>
        </div>
      ) : dataset.length === 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>No data available</CardTitle>
            <CardDescription>
              No processed data found. Try uploading and processing some files first.
            </CardDescription>
          </CardHeader>
        </Card>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Dataset JSON</CardTitle>
              <CardDescription>
                Raw data from processed_data.json
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-gray-100 p-4 rounded-md overflow-auto max-h-[600px]">
                {formattedDataset}
              </pre>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}