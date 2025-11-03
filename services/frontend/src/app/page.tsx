"use client";

import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { DocumentsView } from "@/components/documents/DocumentsView";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { AuthDialog } from "@/components/auth/AuthDialog";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  BookOpen,
  Sparkles,
  BarChart3,
  FileText,
  Brain,
  TrendingUp,
  Users,
} from "lucide-react";

function LandingPage() {
  const [authDialogOpen, setAuthDialogOpen] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "register">("register");

  const handleGetStarted = () => {
    setAuthMode("register");
    setAuthDialogOpen(true);
  };

  const handleSignIn = () => {
    setAuthMode("login");
    setAuthDialogOpen(true);
  };

  const features = [
    {
      icon: Brain,
      title: "AI-Powered Analysis",
      description: "Advanced sentiment analysis and emotional tone detection",
    },
    {
      icon: BarChart3,
      title: "Rich Statistics",
      description:
        "Vocabulary metrics, readability scores, and n-gram analysis",
    },
    {
      icon: FileText,
      title: "Multi-Format Support",
      description: "Process TXT, PDF, DOCX, RTF, and legacy DOC files",
    },
    {
      icon: TrendingUp,
      title: "Real-time Processing",
      description: "Instant analysis with live progress updates",
    },
  ];

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      {/* Hero Section */}
      <div className="mx-auto max-w-6xl px-6 py-24 md:py-32">
        <div className="text-center">
          {/* Main Headline */}
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-semibold mb-8 leading-tight">
            Analyze Literary Texts with{" "}
            <span className="text-violet-600">AI-Powered NLP</span>
          </h1>

          {/* Subheadline */}
          <p className="text-xl text-gray-600 mb-16 max-w-3xl mx-auto leading-relaxed">
            Upload documents, extract insights, and discover patterns in
            literature and historical texts. Perfect for researchers, students,
            and digital humanities enthusiasts.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-6 justify-center mb-24">
            <Button
              size="lg"
              onClick={handleGetStarted}
              className="bg-violet-600 hover:bg-violet-700 text-white px-10 py-4 text-lg font-medium"
            >
              <Sparkles className="mr-2 h-5 w-5" />
              Get Started Free
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={handleSignIn}
              className="border-gray-300 text-gray-700 hover:bg-gray-50 px-10 py-4 text-lg font-medium"
            >
              <Users className="mr-2 h-5 w-5" />
              Sign In
            </Button>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="mx-auto max-w-7xl px-6 pb-24">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature) => (
            <Card
              key={feature.title}
              className="p-8 text-center bg-white border border-gray-200 hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
            >
              <div className="mb-6">
                <feature.icon className="h-12 w-12 mx-auto text-violet-600" />
              </div>
              <h3 className="font-semibold text-xl mb-4 text-gray-900">{feature.title}</h3>
              <p className="text-gray-600 leading-relaxed">
                {feature.description}
              </p>
            </Card>
          ))}
        </div>
      </div>

      {/* Auth Dialog */}
      <AuthDialog
        open={authDialogOpen}
        onOpenChange={setAuthDialogOpen}
        defaultMode={authMode}
      />
    </div>
  );
}

export default function Home() {
  return (
    <ProtectedRoute fallback={<LandingPage />}>
      <DocumentsView />
    </ProtectedRoute>
  );
}
