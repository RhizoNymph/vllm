'use client';

// STUB (standalone port): replaces Neuronpedia's app-wide GlobalProvider
// (models from the DB, auth'd user, toasts, etc.) with a static single-model
// context. Export names/signatures match what the ported jlens components
// import:
//   - globalModels[modelId]?.instruct / ?.layers
//   - user?.name
//   - showToastMessage(ReactNode)
//   - getInferenceEnabledModels() / getInferenceEnabledForModel(modelId)
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';

export interface GlobalModel {
  instruct: boolean;
  displayName: string;
  displayNameShort: string;
  layers: number;
  inferenceEnabled: boolean;
}

const GLOBAL_MODELS: { [key: string]: GlobalModel } = {
  'glm-5.2': {
    instruct: true,
    displayName: 'GLM-5.2',
    displayNameShort: 'GLM-5.2',
    layers: 78,
    inferenceEnabled: true,
  },
};

interface GlobalContextShape {
  globalModels: { [key: string]: GlobalModel };
  user: { name?: string | null } | undefined;
  showToastMessage: (message: ReactNode) => void;
  getInferenceEnabledModels: () => string[];
  getInferenceEnabledForModel: (modelId: string) => boolean;
}

const GlobalContext = createContext<GlobalContextShape | null>(null);

export function useGlobalContext(): GlobalContextShape {
  const ctx = useContext(GlobalContext);
  if (!ctx) {
    throw new Error('useGlobalContext must be used within GlobalProvider');
  }
  return ctx;
}

const TOAST_DURATION_MS = 3500;

export function GlobalProvider({ children }: { children: ReactNode }) {
  const [toast, setToast] = useState<ReactNode | null>(null);
  const toastTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showToastMessage = useCallback((message: ReactNode) => {
    setToast(message);
    if (toastTimeoutRef.current) {
      clearTimeout(toastTimeoutRef.current);
    }
    toastTimeoutRef.current = setTimeout(() => setToast(null), TOAST_DURATION_MS);
  }, []);

  useEffect(
    () => () => {
      if (toastTimeoutRef.current) {
        clearTimeout(toastTimeoutRef.current);
      }
    },
    [],
  );

  const getInferenceEnabledForModel = useCallback((modelId: string) => !!GLOBAL_MODELS[modelId]?.inferenceEnabled, []);
  const getInferenceEnabledModels = useCallback(
    () => Object.keys(GLOBAL_MODELS).filter((m) => GLOBAL_MODELS[m].inferenceEnabled),
    [],
  );

  return (
    <GlobalContext.Provider
      // eslint-disable-next-line react/jsx-no-constructed-context-values
      value={{
        globalModels: GLOBAL_MODELS,
        user: undefined,
        showToastMessage,
        getInferenceEnabledModels,
        getInferenceEnabledForModel,
      }}
    >
      {children}
      {toast != null && (
        <div className="pointer-events-none fixed bottom-6 left-1/2 z-[100] -translate-x-1/2">
          <div className="pointer-events-auto max-w-md rounded-lg border border-slate-200 bg-white px-4 py-3 text-xs text-slate-600 shadow-lg">
            {toast}
          </div>
        </div>
      )}
    </GlobalContext.Provider>
  );
}
