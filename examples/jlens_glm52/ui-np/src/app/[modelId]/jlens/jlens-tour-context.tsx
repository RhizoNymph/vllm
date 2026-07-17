'use client';

// STUB (standalone port): the upstream file re-exports the `DriveStep` type
// from driver.js and shares the tour's currently-highlighted step. The guided
// tour is not ported, so this stub keeps the same exports (context + hook)
// without the driver.js dependency. The context value is always `null`
// (= no tour running), which is exactly how consumers behave outside a tour.
import { createContext, useContext } from 'react';

// Minimal structural stand-in for driver.js's `DriveStep`: consumers only ever
// read `step?.element` (string DOM selector) in the ported components.
export interface DriveStep {
  element?: string | Element | (() => Element);
  [key: string]: unknown;
}

// The step driver.js is currently spotlighting, or `null` when no tour is
// running.
export const JlensTourStepContext = createContext<DriveStep | null>(null);

export function useJlensTourStep(): DriveStep | null {
  return useContext(JlensTourStepContext);
}
