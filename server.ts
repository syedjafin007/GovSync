import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';

dotenv.config();

let aiClient: GoogleGenAI | null = null;

function getGenAI(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY || '';
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return aiClient;
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '10mb' }));

  // API: Health check
  app.get('/api/health', (_req, res) => {
    res.json({
      status: 'ok',
      hasApiKey: Boolean(process.env.GEMINI_API_KEY),
      service: 'GovSync AI Server',
    });
  });

  // API: Multi-turn chat with Gemini and Google Search Grounding
  app.post('/api/gemini/chat', async (req, res) => {
    try {
      const {
        messages,
        systemInstruction,
        model = 'gemini-3.5-flash',
        useSearchGrounding = false,
      } = req.body;

      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        return res.status(400).json({ error: 'Messages array is required.' });
      }

      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        return res.status(500).json({
          error: 'GEMINI_API_KEY is not configured on the server. Please add your API key in Settings > Secrets.',
        });
      }

      // Convert conversation messages to Gemini format
      const formattedContents = messages.map((m: { role: string; content: string }) => ({
        role: m.role === 'assistant' || m.role === 'model' ? 'model' : 'user',
        parts: [{ text: m.content }],
      }));

      const ai = getGenAI();

      // Configure tools and system instruction
      const config: any = {};
      if (systemInstruction) {
        config.systemInstruction = systemInstruction;
      }

      // Only attach googleSearch tool if search grounding is requested
      // gemini-3.5-flash and gemini-3.8-flash support googleSearch
      if (useSearchGrounding) {
        config.tools = [{ googleSearch: {} }];
      }

      // Call Gemini API
      const response = await ai.models.generateContent({
        model,
        contents: formattedContents,
        config,
      });

      const responseText = response.text || '';

      // Extract Grounding Chunks and Web Queries
      const candidate = response.candidates?.[0];
      const groundingMetadata = candidate?.groundingMetadata;
      const groundingChunks = groundingMetadata?.groundingChunks || [];
      const searchQueries = groundingMetadata?.webSearchQueries || [];

      interface WebSource {
        uri: string;
        title: string;
      }

      const sources: WebSource[] = [];
      for (const chunk of groundingChunks as any[]) {
        if (chunk?.web?.uri) {
          sources.push({
            uri: chunk.web.uri,
            title: chunk.web.title || chunk.web.uri,
          });
        }
      }

      return res.json({
        reply: responseText,
        sources,
        searchQueries,
        modelUsed: model,
        grounded: Boolean(sources.length > 0 || searchQueries.length > 0),
      });
    } catch (error: any) {
      console.error('Gemini chat API error:', error);
      const message = error?.message || 'Failed to generate response from Gemini.';
      const status = error?.status || 500;
      return res.status(status).json({
        error: message,
        details: error?.toString(),
      });
    }
  });

  // Mount Vite middleware in development
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`GovSync AI Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
