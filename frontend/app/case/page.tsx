"use client";

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter,
} from '@/components/ui/card';

export default function CasePage() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      },
    },
  };
  
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 100,
        damping: 10,
      },
    },
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-8 bg-gradient-to-b from-gray-900 to-gray-800">
      <motion.div 
        className="z-10 max-w-6xl w-full"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <motion.h1 
          className="text-4xl font-bold mb-6 text-white"
          initial={{ y: -30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          Case Files: The Missing Document Mystery
        </motion.h1>
        
        <motion.div variants={containerVariants} initial="hidden" animate="visible">
          <motion.div variants={itemVariants}>
            <Card className="bg-gray-800 border-gray-700 shadow-lg mb-8">
              <CardHeader>
                <CardTitle className="text-white">Case Overview</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-gray-300">
                  A critical document containing sensitive information has disappeared from a secure database. 
                  The access logs indicate that the files were accessed using Administrator credentials, but 
                  no authorization for transfer was recorded in our systems.
                </p>
                
                <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                  <h3 className="text-xl font-medium mb-2 text-white">Missing Files</h3>
                  <ul className="list-disc pl-6 text-gray-300">
                    <li>Quantum Key Distribution Technical Specifications (QKD-TS-2023)</li>
                    <li>Nexus Architecture Design Document (NADD-v2.3)</li>
                    <li>Implementation Timeline and Milestones (ITM-Q2-2023)</li>
                    <li>Budget Allocation Spreadsheet (BAS-2023-Q2Q4)</li>
                  </ul>
                </div>
                
                <motion.div 
                  className="bg-red-900/30 p-4 rounded-lg border border-red-700"
                  whileHover={{ scale: 1.02 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 10 }}
                >
                  <h3 className="text-xl font-medium mb-2 text-white">Security Alert</h3>
                  <p className="text-gray-300">
                    All external access to the database has been temporarily locked down.
                    The IT security team is conducting a thorough investigation.
                  </p>
                </motion.div>
              </CardContent>
            </Card>
          </motion.div>
          
          <motion.div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <motion.div variants={itemVariants}>
              <Card className="bg-gray-800 border-gray-700 shadow-lg h-full">
                <CardHeader>
                  <CardTitle className="text-white">Evidence Collection</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {[
                    {
                      title: "Security Access Logs",
                      desc: "Detailed logs of system access on April 14-15, 2023",
                      link: "/case/evidence/logs"
                    },
                    {
                      title: "Email Communications",
                      desc: "Email correspondence between team members",
                      link: "/case/evidence/emails"
                    },
                    {
                      title: "System Configuration",
                      desc: "Database and access control configurations",
                      link: "/case/evidence/system"
                    }
                  ].map((item, index) => (
                    <motion.div 
                      key={index}
                      className="bg-gray-900 p-4 rounded-lg border border-gray-700 hover:border-blue-500 transition-all"
                      whileHover={{ 
                        scale: 1.03,
                        boxShadow: "0 0 8px rgba(59, 130, 246, 0.5)"
                      }}
                    >
                      <h3 className="text-xl font-medium mb-2 text-white">{item.title}</h3>
                      <p className="text-gray-300 mb-3">{item.desc}</p>
                      <Link href={item.link}>
                        <Button variant="blue" size="sm">
                          View {item.title.split(' ')[0]}
                        </Button>
                      </Link>
                    </motion.div>
                  ))}
                </CardContent>
              </Card>
            </motion.div>
            
            <motion.div variants={itemVariants}>
              <Card className="bg-gray-800 border-gray-700 shadow-lg h-full">
                <CardHeader>
                  <CardTitle className="text-white">Persons of Interest</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {[
                    {
                      name: "Dr. Alex Rodriguez",
                      role: "Quantum Physics Specialist",
                      background: "Recently onboarded, completed standard background checks",
                      access: "Level 4 (Full access to Quantum systems)"
                    },
                    {
                      name: "Maya Williams",
                      role: "Network Security Engineer",
                      background: "Recently onboarded, completed standard background checks",
                      access: "Level 3 (Network infrastructure access)"
                    },
                    {
                      name: "Thomas Greene",
                      role: "Financial Analyst",
                      background: "Recently onboarded, completed standard background checks",
                      access: "Level 2 (Financial data access only)"
                    }
                  ].map((person, index) => (
                    <motion.div 
                      key={index}
                      className="bg-gray-900 p-4 rounded-lg border border-gray-700"
                      whileHover={{ scale: 1.02 }}
                      transition={{ type: 'spring', stiffness: 400, damping: 10 }}
                    >
                      <h3 className="text-xl font-medium mb-2 text-white">{person.name}</h3>
                      <p className="text-gray-400">Role: {person.role}</p>
                      <p className="text-gray-400">Background: {person.background}</p>
                      <p className="text-gray-400">Access Level: {person.access}</p>
                    </motion.div>
                  ))}
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>
          
          <motion.div variants={itemVariants}>
            <Card className="bg-gray-800 border-gray-700 shadow-lg">
              <CardHeader>
                <CardTitle className="text-white">Investigation Status</CardTitle>
              </CardHeader>
              <CardContent>
                <motion.div 
                  className="bg-yellow-800/30 p-4 rounded-lg border border-yellow-700 mb-4"
                  whileHover={{ scale: 1.01 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 10 }}
                >
                  <p className="text-gray-300">
                    <span className="font-semibold">Status:</span> In Progress
                  </p>
                  <p className="text-gray-300">
                    <span className="font-semibold">Investigation Lead:</span> Detective Agent
                  </p>
                  <p className="text-gray-300">
                    <span className="font-semibold">Team:</span> Forensic Analyst, Researcher, Profiler
                  </p>
                </motion.div>
              </CardContent>
              <CardFooter>
                <Link href="/case/solve">
                  <Button 
                    variant="green" 
                    className="w-full group relative overflow-hidden"
                  >
                    <span className="relative z-10">Run Investigation</span>
                    <span className="absolute inset-0 bg-gradient-to-r from-green-600 to-green-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
                  </Button>
                </Link>
              </CardFooter>
            </Card>
          </motion.div>
        </motion.div>
      </motion.div>
    </main>
  );
} 