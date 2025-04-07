"use client";

import React from 'react';

export default function AboutPage() {
  return (
    <main className="flex min-h-screen flex-col items-center p-8 bg-gradient-to-b from-gray-900 to-gray-800">
      <div className="z-10 max-w-5xl w-full">
        <h1 className="text-4xl font-bold mb-6 text-white">About Autonomous Document Intelligence</h1>
        
        <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700 mb-8">
          <h2 className="text-2xl font-semibold mb-4 text-white">Our Mission</h2>
          <p className="text-gray-300 mb-4">
            Autonomous Document Intelligence is a cutting-edge platform designed to transform unstructured and 
            semi-structured documents into structured, actionable intelligence. Our mission is to help enterprises 
            extract valuable insights from their documents using advanced AI agents that work together to solve 
            complex document processing challenges.
          </p>
          <p className="text-gray-300">
            Traditional document processing systems are rigid and require extensive configuration. Our approach 
            uses natural language commands to dynamically adapt to your specific document processing needs, 
            making it more flexible, powerful, and user-friendly.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700">
            <h2 className="text-2xl font-semibold mb-4 text-white">Key Features</h2>
            <ul className="space-y-4">
              <li className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                <h3 className="text-lg font-medium mb-1 text-white">Natural Language Control</h3>
                <p className="text-gray-400">Process documents using simple conversational commands</p>
              </li>
              <li className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                <h3 className="text-lg font-medium mb-1 text-white">Multimodal Processing</h3>
                <p className="text-gray-400">Handle text, images, audio, and video in a unified system</p>
              </li>
              <li className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                <h3 className="text-lg font-medium mb-1 text-white">Dynamic Schema Adaptation</h3>
                <p className="text-gray-400">Automatically adapt to changing document structures</p>
              </li>
              <li className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                <h3 className="text-lg font-medium mb-1 text-white">Continuous Learning</h3>
                <p className="text-gray-400">Improve over time based on user feedback and interactions</p>
              </li>
              <li className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                <h3 className="text-lg font-medium mb-1 text-white">Multi-Agent Architecture</h3>
                <p className="text-gray-400">Specialized agents collaborate to solve complex problems</p>
              </li>
            </ul>
          </div>
          
          <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700">
            <h2 className="text-2xl font-semibold mb-4 text-white">How It Works</h2>
            <div className="space-y-4">
              <div className="relative pl-8 pb-8 border-l-2 border-blue-600">
                <div className="absolute w-4 h-4 bg-blue-600 rounded-full -left-[9px]"></div>
                <h3 className="text-lg font-medium mb-1 text-white">Document Ingestion</h3>
                <p className="text-gray-400">Documents of any type are uploaded and prepared for processing</p>
              </div>
              
              <div className="relative pl-8 pb-8 border-l-2 border-green-600">
                <div className="absolute w-4 h-4 bg-green-600 rounded-full -left-[9px]"></div>
                <h3 className="text-lg font-medium mb-1 text-white">Natural Language Instruction</h3>
                <p className="text-gray-400">You specify what you want to extract or analyze in plain English</p>
              </div>
              
              <div className="relative pl-8 pb-8 border-l-2 border-yellow-600">
                <div className="absolute w-4 h-4 bg-yellow-600 rounded-full -left-[9px]"></div>
                <h3 className="text-lg font-medium mb-1 text-white">Multi-Agent Processing</h3>
                <p className="text-gray-400">Our team of specialized agents work together:</p>
                <ul className="list-disc pl-6 text-gray-400">
                  <li>Detective coordinates the investigation</li>
                  <li>Forensic Analyst processes multimedia content</li>
                  <li>Researcher gathers contextual information</li>
                  <li>Profiler analyzes patterns and behaviors</li>
                </ul>
              </div>
              
              <div className="relative pl-8">
                <div className="absolute w-4 h-4 bg-purple-600 rounded-full -left-[9px]"></div>
                <h3 className="text-lg font-medium mb-1 text-white">Structured Output</h3>
                <p className="text-gray-400">Results are delivered in organized, structured formats</p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700">
          <h2 className="text-2xl font-semibold mb-4 text-white">Use Cases</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
              <h3 className="text-lg font-medium mb-2 text-white">Financial Analysis</h3>
              <p className="text-gray-400">Extract financial data from annual reports, invoices, and contracts</p>
            </div>
            
            <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
              <h3 className="text-lg font-medium mb-2 text-white">Legal Document Review</h3>
              <p className="text-gray-400">Analyze legal documents for key clauses, entities, and obligations</p>
            </div>
            
            <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
              <h3 className="text-lg font-medium mb-2 text-white">Healthcare Records</h3>
              <p className="text-gray-400">Process medical records to extract diagnoses, treatments, and outcomes</p>
            </div>
            
            <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
              <h3 className="text-lg font-medium mb-2 text-white">Media Transcription</h3>
              <p className="text-gray-400">Convert audio and video recordings into searchable text with analysis</p>
            </div>
            
            <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
              <h3 className="text-lg font-medium mb-2 text-white">Customer Feedback</h3>
              <p className="text-gray-400">Analyze customer surveys and feedback for sentiment and key insights</p>
            </div>
            
            <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
              <h3 className="text-lg font-medium mb-2 text-white">Research & Development</h3>
              <p className="text-gray-400">Organize and extract insights from research papers and technical documents</p>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700 mt-8">
          <h2 className="text-2xl font-semibold mb-4 text-white">Technology Stack</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-xl font-medium mb-3 text-white">Backend</h3>
              <ul className="space-y-2 text-gray-300">
                <li><span className="font-semibold">CrewAI:</span> Multi-agent orchestration framework</li>
                <li><span className="font-semibold">Python:</span> Core backend processing</li>
                <li><span className="font-semibold">OpenAI:</span> LLM integration for agent capabilities</li>
                <li><span className="font-semibold">PyTesseract:</span> OCR text extraction</li>
                <li><span className="font-semibold">OpenCV:</span> Image analysis</li>
                <li><span className="font-semibold">Whisper:</span> Audio transcription</li>
              </ul>
            </div>
            <div>
              <h3 className="text-xl font-medium mb-3 text-white">Frontend</h3>
              <ul className="space-y-2 text-gray-300">
                <li><span className="font-semibold">Next.js:</span> React framework</li>
                <li><span className="font-semibold">TypeScript:</span> Type-safe JavaScript</li>
                <li><span className="font-semibold">Tailwind CSS:</span> Utility-first styling</li>
                <li><span className="font-semibold">React Dropzone:</span> File upload interface</li>
                <li><span className="font-semibold">Axios:</span> HTTP client for API requests</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
} 