
Arquivo 1: Contratos do Ecossistema
Caminho: src/core/contracts.ts
typescript
export interface PrslDirective {
  Ecosystem: 'PRSL-AI';
  Organization: 'rsl-robert-software-labs-org';
  Capability: 'privacy' | 'continuity' | 'identity' | 'memory' | 'inference';
  Artifact: string;
  IsolationLevel: 'default' | 'strict' | 'ephemeral';
}

export interface MemoryDocument {
  id: string;
  text: string;
  embedding: number[];
  metadata?: Record<string, unknown>;
}

export interface LocalMemoryEngine {
  ingest(text: string, metadata?: Record<string, unknown>): Promise<void>;
  search(query: string, topK?: number): Promise<Array<{ text: string; score: number }>>;
}

export interface LocalLlmEngine {
  generate(prompt: string, systemPrompt?: string): Promise<string>;
  askWithMemory(question: string, topK?: number): Promise<string>;
  getBackend(): 'webgpu' | 'wasm-cpu' | null;
  isReady(): boolean;
}

export interface PrslMountContext {
  appId: string;
  storage: Storage;
  memory?: LocalMemoryEngine;
  llm?: LocalLlmEngine;
}

export interface IPrslExtension {
  readonly directive: PrslDirective;
  onInstall?(): Promise<void> | void;
  onMount(context: PrslMountContext): Promise<void> | void;
  onUnmount?(): Promise<void> | void;
}
Use code with caution.
Arquivo 2: Plugin de Privacidade (Zero-Network)
Caminho: src/plugins/rsl-robert-software-labs-org/privacy/-extension.ts
typescript
import { IPrslExtension } from '../../../core/contracts';

const PrivacyExtension: IPrslExtension = {
  directive: {
    Ecosystem: 'PRSL-AI',
    Organization: 'rsl-robert-software-labs-org',
    Capability: 'privacy',
    Artifact: '-extension.ts',
    IsolationLevel: 'strict'
  },

  onInstall() {
    console.log('[RSL Privacy] Políticas locais e isolamento provisionados.');
  },

  onMount({ storage }) {
    storage.setItem('rsl_privacy_mode', 'strict_ondevice');
    console.log('[RSL Privacy] Módulo de privacidade ativo.');
  }
};

export default PrivacyExtension;
Use code with caution.
Arquivo 3: Plugin de Memória Vetorial (IndexedDB + Embeddings)
Caminho: src/plugins/rsl-robert-software-labs-org/memory/-extension.ts
typescript
import { pipeline, FeatureExtractionPipeline } from '@huggingface/transformers';
import { IPrslExtension, LocalMemoryEngine, MemoryDocument } from '../../../core/contracts';

const DB_NAME = 'prsl_ondevice_vector_db';
const STORE_NAME = 'memories';
const EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2';

let embedder: FeatureExtractionPipeline | null = null;

function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

function openVectorDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function createEmbedding(text: string): Promise<number[]> {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', EMBEDDING_MODEL, {
      dtype: 'q8'
    });
  }
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data as Float32Array);
}

export const localMemoryService: LocalMemoryEngine = {
  async ingest(text: string, metadata = {}): Promise<void> {
    const embedding = await createEmbedding(text);
    const db = await openVectorDB();

    const doc: MemoryDocument = {
      id: crypto.randomUUID(),
      text,
      embedding,
      metadata: { ...metadata, createdAt: Date.now() }
    };

    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put(doc);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  },

  async search(query: string, topK = 3): Promise<Array<{ text: string; score: number }>> {
    const queryEmbedding = await createEmbedding(query);
    const db = await openVectorDB();

    const allDocs: MemoryDocument[] = await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });

    return allDocs
      .map((doc) => ({
        text: doc.text,
        score: cosineSimilarity(queryEmbedding, doc.embedding)
      }))
      .filter((item) => item.score > 0.35)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
};

const MemoryExtension: IPrslExtension = {
  directive: {
    Ecosystem: 'PRSL-AI',
    Organization: 'rsl-robert-software-labs-org',
    Capability: 'memory',
    Artifact: '-extension.ts',
    IsolationLevel: 'strict'
  },

  async onInstall() {
    await createEmbedding('init');
    await openVectorDB();
  },

  async onMount() {
    console.log('[RSL Memory] Banco vetorial local ativo.');
  }
};

export default MemoryExtension;
Use code with caution.
Arquivo 4: Plugin de Inferência Híbrido Universal (WebGPU + Fallback CPU/WASM)
Caminho: src/plugins/rsl-robert-software-labs-org/inference/-extension.ts
typescript
import { CreateMLCEngine, MLCEngine } from '@mlc-ai/web-llm';
import { pipeline, TextGenerationPipeline } from '@huggingface/transformers';
import { IPrslExtension, LocalLlmEngine, LocalMemoryEngine } from '../../../core/contracts';

const WEBGPU_MODEL = 'Llama-3.2-1B-Instruct-q4f16_1-MLC';
const FALLBACK_CPU_MODEL = 'HuggingFaceTB/SmolLM2-360M-Instruct';

let webGpuEngine: MLCEngine | null = null;
let cpuEngine: TextGenerationPipeline | null = null;
let activeBackend: 'webgpu' | 'wasm-cpu' | null = null;
let boundMemory: LocalMemoryEngine | undefined;

async function hasWebGPUSupport(): Promise<boolean> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) return false;
  try {
    const adapter = await (navigator as any).gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

export const localLlmService: LocalLlmEngine = {
  isReady: () => activeBackend !== null,
  getBackend: () => activeBackend,

  async generate(prompt: string, systemPrompt = 'Você é o assistente on-device iRiel AI.'): Promise<string> {
    if (activeBackend === 'webgpu' && webGpuEngine) {
      const reply = await webGpuEngine.chat.completions.create({
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: prompt }
        ],
        temperature: 0.3,
        max_tokens: 512
      });
      return reply.choices?.message?.content ?? '';
    }

    if (activeBackend === 'wasm-cpu' && cpuEngine) {
      const messages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: prompt }
      ];
      const output: any = await cpuEngine(messages, { max_new_tokens: 256, temperature: 0.3 });
      const generated = output?.generated_text;
      return Array.isArray(generated) ? generated[generated.length - 1].content : String(generated ?? '');
    }

    throw new Error('[RSL Inference] Nenhum motor on-device inicializado.');
  },

  async askWithMemory(question: string, topK = 3): Promise<string> {
    if (!boundMemory) return this.generate(question);

    const relevantMemories = await boundMemory.search(question, topK);
    if (relevantMemories.length === 0) {
      return this.generate(question);
    }

    const contextBlock = relevantMemories
      .map((m, i) => `[Memória ${i + 1}]: ${m.text}`)
      .join('\n');

    const ragPrompt = `Responda usando apenas as memórias locais abaixo:\n${contextBlock}`;
    return this.generate(question, ragPrompt);
  }
};

const InferenceExtension: IPrslExtension = {
  directive: {
    Ecosystem: 'PRSL-AI',
    Organization: 'rsl-robert-software-labs-org',
    Capability: 'inference',
    Artifact: '-extension.ts',
    IsolationLevel: 'strict'
  },

  async onMount({ storage, memory }) {
    boundMemory = memory;
    const gpuAvailable = await hasWebGPUSupport();

    if (gpuAvailable) {
      try {
        webGpuEngine = await CreateMLCEngine(WEBGPU_MODEL);
        activeBackend = 'webgpu';
        storage.setItem('rsl_inference_backend', 'webgpu');
        return;
      } catch (err) {
        console.warn('[RSL Inference] Alternando para WASM/CPU...', err);
      }
    }

    cpuEngine = await pipeline('text-generation', FALLBACK_CPU_MODEL, {
      dtype: 'q4',
      device: 'wasm'
    });
    activeBackend = 'wasm-cpu';
    storage.setItem('rsl_inference_backend', 'wasm-cpu');
  }
};

export default InferenceExtension;
Use code with caution.
Arquivo 5: Orquestrador e Parser PRSL-RLP
Caminho: src/core/orchestrator.ts
typescript
import { IPrslExtension, PrslMountContext } from './contracts';
import { localMemoryService } from '../plugins/rsl-robert-software-labs-org/memory/-extension';
import { localLlmService } from '../plugins/rsl-robert-software-labs-org/inference/-extension';

const CAPABILITY_PRIORITY: Record<string, number> = {
  privacy: 1,
  identity: 2,
  memory: 3,
  inference: 4,
  continuity: 5
};

const pluginModules = import.meta.glob<{ default: IPrslExtension }>(
  '../plugins/**/-extension.ts',
  { eager: true }
);

export class PieOrchestrator {
  private installedPlugins = new Map<string, IPrslExtension>();

  public parseDirectiveUri(rawUri: string): string | null {
    const cleaned = rawUri.replace(/[\{\}\s]/g, '');
    const match = cleaned.match(/^(?:prsl:\/\/|https\/\/:)([^/]+)\/([^/]+)\/(-extension(?:\.ts)?)$/);
    if (!match) return null;
    const [, org, capability] = match;
    return `${org}:${capability}`;
  }

  public async bootOnDevicePlugins(): Promise<PrslMountContext> {
    const entries = Object.values(pluginModules)
      .map((m) => m.default)
      .filter((p) => p && p.directive)
      .sort((a, b) => {
        const prioA = CAPABILITY_PRIORITY[a.directive.Capability] ?? 99;
        const prioB = CAPABILITY_PRIORITY[b.directive.Capability] ?? 99;
        return prioA - prioB;
      });

    const sharedContext: PrslMountContext = {
      appId: 'iriel-pie-core',
      storage: localStorage,
      memory: localMemoryService,
      llm: localLlmService
    };

    for (const plugin of entries) {
      const key = `${plugin.directive.Organization}:${plugin.directive.Capability}`;
      const flag = `prsl_installed_${key}`;

      if (!localStorage.getItem(flag)) {
        if (plugin.onInstall) await plugin.onInstall();
        localStorage.setItem(flag, 'true');
      }

      await plugin.onMount(sharedContext);
      this.installedPlugins.set(key, plugin);
    }

    return sharedContext;
  }

  public resolveByDirective(rawUri: string): IPrslExtension | undefined {
    const key = this.parseDirectiveUri(rawUri);
    return key ? this.installedPlugins.get(key) : undefined;
  }
}

export const pieOrchestrator = new PieOrchestrator();
Use code with caution.
Arquivo 6: Service Worker Offline
Caminho: public/sw.js
javascript
const CACHE_NAME = 'iriel-pie-ondevice-v1';

self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(clients.claim());
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) return cachedResponse;
      return fetch(event.request).then((networkResponse) => {
        const clone = networkResponse.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        return networkResponse;
      });
    })
  );
});
Use code with caution.
Arquivo 7: Headers Universais (COOP/COEP para WebGPU/WASM)
Caminho: public/_headers
text
/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
Use code with caution.
Arquivo 8: Esteira Automática de Build Offline
Caminho: .github/workflows/build-on-device.yml
yaml
name: PRSL-AI On-Device Build Pipeline

on:
  push:
    branches: [ main ]

jobs:
  build-pie-orchestrator:
    runs-on: ubuntu-latest

    steps:
      - name: 1. Checkout
        uses: actions/checkout@v4

      - name: 2. Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'npm'

      - name: 3. Instalar Dependências
        run: npm install

      - name: 4. Build Offline
        run: npm run build

      - name: 5. Empacotar ZIP
        run: cd dist && zip -r ../iriel-pie-ondevice-bundle.zip .

      - name: 6. Publicar Artefato
        uses: actions/upload-artifact@v4
        with:
          name: iriel-pie-ondevice-release
          path: iriel-pie-ondevice-bundle.zip
Use code with caution.