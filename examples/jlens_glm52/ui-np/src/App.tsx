import { JlensTourStepContext } from '@/app/[modelId]/jlens/jlens-tour-context';
import { LensMode, LensModeContext, LensModeSetContext } from '@/components/jlens/jlens-lens-mode';
import JlensPanel from '@/components/jlens/jlens-panel';
import { GlobalProvider } from '@/components/provider/global-provider';
import { useState } from 'react';

// The single model this standalone build serves. Must match a key in the
// global-provider stub's model map and the `modelId` the sidecar's
// `/api/lens/prompt` endpoint expects.
const MODEL_ID = 'glm-5.2';

// Minimal standalone page shell. Replicates the provider nesting of
// Neuronpedia's jlens-page-client.tsx (tour step + lens mode contexts around
// JlensPanel) without the Next router / share loading / model selector /
// tour / intro-video parts.
export default function App() {
  const [lensMode, setLensMode] = useState<LensMode>(LensMode.JACOBIAN_LENS);

  return (
    <GlobalProvider>
      <JlensTourStepContext.Provider value={null}>
        <LensModeContext.Provider value={lensMode}>
          <LensModeSetContext.Provider value={setLensMode}>
            <div className="mx-auto flex max-h-[100dvh] min-h-[100dvh] w-full flex-col bg-slate-100">
              <div className="flex w-full items-center justify-center px-3 pt-3 sm:px-6 sm:pb-3 sm:pt-6">
                <div className="flex w-full max-w-screen-2xl flex-row items-center justify-between">
                  <div className="flex flex-col items-start justify-center">
                    <div className="whitespace-nowrap text-[14px] font-semibold leading-none text-slate-800 sm:text-base">
                      Jacobian Lens
                    </div>
                    <div className="mt-[3px] whitespace-nowrap text-[10px] text-slate-500 sm:text-[11.5px]">
                      Gurnee et al.
                    </div>
                  </div>
                  <div className="rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600">
                    GLM-5.2
                  </div>
                </div>
              </div>
              <div className="mx-auto flex h-full min-h-0 w-full max-w-screen-xl flex-1 flex-col sm:py-3">
                <JlensPanel key={MODEL_ID} modelId={MODEL_ID} inferenceAvailable />
              </div>
            </div>
          </LensModeSetContext.Provider>
        </LensModeContext.Provider>
      </JlensTourStepContext.Provider>
    </GlobalProvider>
  );
}
