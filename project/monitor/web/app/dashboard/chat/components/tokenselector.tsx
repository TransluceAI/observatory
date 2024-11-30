import React, {
  useState,
  useRef,
  useEffect,
  useMemo,
  useCallback,
} from 'react';
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from '@/components/ui/context-menu';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { ChatToken } from '@/app/types/tokens';
import { Neuron, NeuronsMetadataDict } from '@/app/types/neuronData';
import { SteeringSpec } from '@/app/types/neuronFilters';
import { X } from 'lucide-react'; // Import the X icon for dismissing the message

interface MessageSegment {
  role: 'user' | 'assistant' | 'system' | 'unknown';
  tokens: ChatToken[];
}

interface TokenSelectorProps {
  // Data
  chatTokens: ChatToken[];
  neurons?: Neuron[];
  neuronsMetadataDict?: NeuronsMetadataDict;
  steeringSpecs?: SteeringSpec[];

  // Display modulation
  mousedOverNeurons?: Neuron[];
  showNeuronsFrom?: 'attribution' | 'activation';
  selectedAttributionToken?: number;

  // What happens when the user selects a range of tokens
  initialSelection?: [number, number][];
  onSelectionChange: (ranges: [number, number][]) => void;
  resetSelectionRef?: React.MutableRefObject<(() => void) | null>;

  // Allow multiple non-contiguous selections?
  allowMultipleSelections?: boolean;

  // Highlight a token
  presetTokenHighlight?: PresetFlowTokenHighlight;

  // Misc things
  disableUserSelection?: boolean;
  alwaysTranslucent?: boolean;
  customSelectionClassName?: string;
  ignoreFirstToken?: boolean; // BOS tokens are useless
}

import { cn } from '@/lib/utils';
import { PresetFlowTokenHighlight } from '@/app/types/presetFlow';
import { setMousedOverTokenIndex } from '@/app/store/slices/neuronsSlice';
import { useDispatch } from 'react-redux';

export function TokenSelector({
  chatTokens,
  neurons,
  neuronsMetadataDict,
  steeringSpecs,
  mousedOverNeurons,
  showNeuronsFrom = 'activation',
  selectedAttributionToken,
  initialSelection,
  onSelectionChange,
  resetSelectionRef,
  allowMultipleSelections = false,
  presetTokenHighlight,
  disableUserSelection = false, // New prop with default value
  alwaysTranslucent = false,
  customSelectionClassName,
  ignoreFirstToken = false,
}: TokenSelectorProps) {
  const dispatch = useDispatch();

  const [currentSelection, setCurrentSelection] = useState<
    [number, number, number] | null
  >(null);
  const [selectedRanges, setSelectedRanges] = useState<[number, number][]>(
    initialSelection ?? []
  );
  // const selectedRanges: [number, number][] = useMemo(() => {
  //   return currentSelection ? [[currentSelection[0], currentSelection[2]]] : [];
  // }, [currentSelection]);
  const [localMousedOverTokenIndex, setLocalMousedOverTokenIndex] = useState<
    number | null
  >(null);

  const mousedOverNeuronIdsSet = useMemo(() => {
    if (mousedOverNeurons === undefined) return new Set();
    return new Set(
      mousedOverNeurons.map((neuron) => `${neuron.layer},${neuron.neuron}`)
    );
  }, [mousedOverNeurons]);

  /**
   * Handle mouse events
   */
  const handleMouseDown = (index: number, event: React.MouseEvent) => {
    event.preventDefault();
    setCurrentSelection([index, index, index]);
    setSelectedRanges([]);
  };
  const handleMouseEnter = (index: number) => {
    if (currentSelection) {
      const [start, pivot, end] = currentSelection;
      if (index < pivot) {
        setCurrentSelection([index, pivot, pivot]);
      } else {
        setCurrentSelection([pivot, pivot, index]);
      }
    }
  };
  const handleMouseUp = useCallback(() => {
    if (currentSelection) {
      const newRanges: [number, number][] = [
        [currentSelection[0], currentSelection[2]],
      ];
      setSelectedRanges(newRanges);
      onSelectionChange(newRanges);
      setCurrentSelection(null);
    }
  }, [currentSelection]);

  useEffect(() => {
    const handleGlobalMouseUp = () => {
      handleMouseUp();
    };

    document.addEventListener('mouseup', handleGlobalMouseUp);
    return () => {
      document.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, [handleMouseUp]);

  /**
   * Parse tokens into message segments
   */
  const parseTokensIntoSegments = (tokens: ChatToken[]): MessageSegment[] => {
    const messages: MessageSegment[] = [];
    let currentIndex = 0;
    const tokensLength = tokens.length;

    while (currentIndex < tokensLength) {
      if (tokens[currentIndex].token === '<|start_header_id|>') {
        // Start of a new message segment
        let headerTokens: ChatToken[] = [tokens[currentIndex]];
        currentIndex++;

        // Collect header tokens
        while (
          currentIndex < tokensLength &&
          tokens[currentIndex].token !== '<|end_header_id|>'
        ) {
          headerTokens.push(tokens[currentIndex]);
          currentIndex++;
        }

        // Add end header token if present
        if (
          currentIndex < tokensLength &&
          tokens[currentIndex].token === '<|end_header_id|>'
        ) {
          headerTokens.push(tokens[currentIndex]);
          currentIndex++;
        }

        // Determine role from header tokens
        const roleTokens = headerTokens.slice(1, -1).map((t) => t.token);
        const role = roleTokens.join('').trim() as MessageSegment['role'];

        // Collect message tokens
        let messageTokens: ChatToken[] = [];
        while (
          currentIndex < tokensLength &&
          tokens[currentIndex].token !== '<|eot_id|>' &&
          tokens[currentIndex].token !== '<|start_header_id|>'
        ) {
          messageTokens.push(tokens[currentIndex]);
          currentIndex++;
        }

        // Add EOT token if present
        if (
          currentIndex < tokensLength &&
          tokens[currentIndex].token === '<|eot_id|>'
        ) {
          messageTokens.push(tokens[currentIndex]);
          currentIndex++;
        }

        messages.push({ role, tokens: [...headerTokens, ...messageTokens] });
      } else {
        // Collect tokens until next '<|start_header_id|>' or end
        let unknownTokens: ChatToken[] = [];
        while (
          currentIndex < tokensLength &&
          tokens[currentIndex].token !== '<|start_header_id|>'
        ) {
          unknownTokens.push(tokens[currentIndex]);
          currentIndex++;
        }
        if (unknownTokens.length > 0) {
          messages.push({ role: 'unknown', tokens: unknownTokens });
        }
      }
    }
    return messages;
  };

  const messages = useMemo(
    () => parseTokensIntoSegments(chatTokens),
    [chatTokens]
  );

  /**
   * Calculate maximum displayed values
   */
  const maxDisplayValue = useMemo(() => {
    if (
      neurons === undefined ||
      neuronsMetadataDict === undefined ||
      mousedOverNeurons === undefined
    ) {
      return 0;
    }

    const curNeurons = neurons.filter((neuron) =>
      mousedOverNeuronIdsSet.has(`${neuron.layer},${neuron.neuron}`)
    );

    if (showNeuronsFrom === 'activation') {
      return curNeurons.reduce((max, neuron) => {
        const act =
          neuronsMetadataDict.run[
            `${neuron.layer},${neuron.neuron},${neuron.token}`
          ]?.activation ?? 0;
        return Math.max(max, Math.abs(act));
      }, 0);
    } else {
      return curNeurons.reduce((max, neuron) => {
        const attrs =
          neuronsMetadataDict.run[
            `${neuron.layer},${neuron.neuron},${neuron.token}`
          ]?.attributions;
        if (!attrs) return max;
        const attr =
          attrs[(selectedAttributionToken ?? 0) - 1]?.attribution ?? 0;
        return Math.max(max, Math.abs(attr));
      }, 0);
    }
  }, [
    chatTokens,
    neurons,
    neuronsMetadataDict,
    mousedOverNeuronIdsSet,
    showNeuronsFrom,
  ]);

  /**
   * Summed absolute activities for all tokens
  //  * Only in the token selection if available
   */
  const summedAbsoluteActivities = useMemo(() => {
    if (!neurons || !neuronsMetadataDict) return {};

    const tokenActivities: { [tokenIndex: number]: number } = {};

    for (const neuron of neurons) {
      const tokenIdx = neuron.token;
      if (tokenIdx === null) continue;
      // if (
      //   selectedRanges.length > 0 &&
      //   !selectedRanges.some(
      //     ([start, end]) => tokenIdx >= start && tokenIdx <= end
      //   )
      // )
      //   continue;
      if (!(tokenIdx in tokenActivities)) {
        tokenActivities[tokenIdx] = 0;
      }

      if (showNeuronsFrom === 'activation') {
        const activation =
          neuronsMetadataDict.run[
            `${neuron.layer},${neuron.neuron},${neuron.token}`
          ]?.activation ?? 0;
        tokenActivities[tokenIdx] += Math.abs(activation);
      } else {
        const attrs =
          neuronsMetadataDict.run[
            `${neuron.layer},${neuron.neuron},${neuron.token}`
          ]?.attributions;
        if (attrs) {
          const attribution =
            attrs[(selectedAttributionToken ?? 0) - 1]?.attribution ?? 0;
          tokenActivities[tokenIdx] += Math.abs(attribution);
        }
      }
    }

    return tokenActivities;
  }, [neurons, neuronsMetadataDict, showNeuronsFrom, selectedRanges]);
  const maxSummedAbsoluteActivity = useMemo(() => {
    if (Object.keys(summedAbsoluteActivities).length === 0) return 0;
    return Math.max(...Object.values(summedAbsoluteActivities));
  }, [summedAbsoluteActivities]);

  /**
   * Get customization options for a token
   */
  const getTokenCustomization = (index: number) => {
    // If preset tokens are shown, and this isn't one, quit.
    if (
      presetTokenHighlight &&
      !(
        presetTokenHighlight.indexRange[0] <= index &&
        presetTokenHighlight.indexRange[1] >= index
      )
    ) {
      return {
        className: 'cursor-not-allowed pointer-events-none',
      };
    }

    // Determine token states
    const isSelected =
      selectedRanges.some(([start, end]) => index >= start && index <= end) ||
      (currentSelection &&
        index >= currentSelection[0] &&
        index <= currentSelection[2]);
    const anySelected = selectedRanges.length > 0 || currentSelection !== null;

    const steeringSpecsForToken =
      steeringSpecs !== undefined
        ? steeringSpecs.filter(
            (spec) =>
              spec.isSteering &&
              spec.tokenRanges?.some(
                ([start, end]) => index >= start && index <= end
              )
          )
        : [];

    const curTokenNeurons = neurons?.filter(
      (neuron) =>
        neuron.token === index &&
        mousedOverNeuronIdsSet.has(`${neuron.layer},${neuron.neuron}`)
    );

    const isNeuronMousedOver =
      neurons !== undefined && curTokenNeurons && curTokenNeurons.length > 0;

    const isAttributionSelected = index === selectedAttributionToken;

    let className = 'cursor-pointer ';
    let styleDict: React.CSSProperties = {};
    let tooltipContent: React.ReactNode = null;
    let hasTooltip = false;
    let tooltipOpen = undefined;
    let tooltipContentClassName = undefined;
    let tooltipSide: 'bottom' | 'top' | 'right' | 'left' | undefined =
      undefined;

    /**
     * Apply styles based on states
     */

    // If something other than this token is selected, reduce opacity
    // Don't do this for attribution selection
    if (anySelected && !isSelected && !isAttributionSelected) {
      className += 'text-gray-900 text-opacity-40 bg-opacity-40 ';
    }

    // Borders
    if (isSelected) {
      className += customSelectionClassName ?? 'border-b-2 border-blue-500 ';
    } else if (localMousedOverTokenIndex === index) {
      className += 'border-b-2 border-blue-300 ';
    }
    if (steeringSpecsForToken.length > 0) {
      className += 'border-b-2 border-green-500';
      tooltipContent = `Steered by ${steeringSpecsForToken.map((spec) => `"${spec.name}"`).join(', ')}`;
      tooltipContentClassName = 'text-sm';
      hasTooltip = true;
    }

    // Highlight preset tokens
    if (
      presetTokenHighlight &&
      presetTokenHighlight.indexRange[0] <= index &&
      presetTokenHighlight.indexRange[1] >= index
    ) {
      className += 'bg-yellow-300';

      // Only show tooltip in the middle
      if (index === presetTokenHighlight.indexRange[1]) {
        tooltipContent = presetTokenHighlight.message;
        hasTooltip = true;
        tooltipOpen = true;
        tooltipContentClassName = 'text-base text-center';
        tooltipSide = 'bottom';
      }
    } else if (isAttributionSelected) {
      className += 'bg-orange-300 ';
      // tooltipContent = 'Selected by attribution';
      // hasTooltip = true;
    } else if (isNeuronMousedOver) {
      let displayValue = 0;
      if (showNeuronsFrom === 'activation') {
        displayValue = curTokenNeurons.reduce((sum, neuron) => {
          return (
            sum +
            (neuronsMetadataDict?.run[
              `${neuron.layer},${neuron.neuron},${neuron.token}`
            ]?.activation ?? 0)
          );
        }, 0);
      } else {
        displayValue = curTokenNeurons.reduce((sum, neuron) => {
          const attrs =
            neuronsMetadataDict?.run[
              `${neuron.layer},${neuron.neuron},${neuron.token}`
            ]?.attributions;
          if (!attrs || Object.keys(attrs).length === 0) {
            return sum;
          }
          const firstAttrKey = Number(Object.keys(attrs)[0] as string);
          const firstAttr = attrs[firstAttrKey];
          return sum + (firstAttr.attribution ?? 0);
        }, 0);
      }

      const activationOpacity = Math.abs(displayValue) / maxDisplayValue;

      if (displayValue > 0) {
        styleDict = {
          backgroundColor: `rgba(134, 239, 172, ${activationOpacity})`, // green-300 if pos
        };
      } else {
        styleDict = {
          backgroundColor: `rgba(252, 165, 165, ${activationOpacity})`, // red-300 if neg
        };
      }
    } else if (mousedOverNeurons == undefined) {
      const tokenActivity = summedAbsoluteActivities[index] || 0;
      const activityRatio = tokenActivity / maxSummedAbsoluteActivity;
      if (activityRatio > 0) {
        const opacity = Math.min(activityRatio, 0.8); // Cap opacity at 0.8
        styleDict = {
          backgroundColor: `rgba(216, 180, 254, ${opacity})`, // purple-300
        };
      }
    }

    return {
      className,
      styleDict,
      hasTooltip,
      tooltipContent,
      tooltipOpen,
      tooltipContentClassName,
      tooltipSide,
    };
  };

  const resetSelection = () => {
    setCurrentSelection(null);
    setSelectedRanges([]);
  };

  useEffect(() => {
    if (resetSelectionRef) {
      resetSelectionRef.current = resetSelection;
    }
  }, [resetSelectionRef]);

  const tokenCustomizations = useMemo(() => {
    return chatTokens.map((_, index) => getTokenCustomization(index));
  }, [
    chatTokens,
    selectedRanges,
    currentSelection,
    steeringSpecs,
    mousedOverNeurons,
    neurons,
    neuronsMetadataDict,
    showNeuronsFrom,
    selectedAttributionToken,
    presetTokenHighlight,
    localMousedOverTokenIndex,
    maxDisplayValue,
    summedAbsoluteActivities,
    maxSummedAbsoluteActivity,
  ]);

  let globalTokenIndex = 0;
  return (
    <>
      <div
        className={cn(
          'select-none w-full space-y-2',
          disableUserSelection && 'pointer-events-none'
        )}
      >
        {messages.map((message, messageIndex) => {
          const isUserOrSystemOrUnknown =
            message.role === 'user' ||
            message.role === 'system' ||
            message.role === 'unknown';
          const messageClassName = isUserOrSystemOrUnknown
            ? 'flex justify-end'
            : 'flex justify-start';
          const bubbleColorClass =
            mousedOverNeurons || alwaysTranslucent
              ? 'bg-gray-100'
              : isUserOrSystemOrUnknown
                ? 'bg-[#dae4fbff]'
                : 'bg-gray-200';
          const bubbleClassName = cn(
            'max-w-[80%] p-2 rounded-lg shadow-sm transition-colors duration-150 inline-flex flex-wrap',
            bubbleColorClass
          );
          return (
            <div key={messageIndex} className={messageClassName}>
              <div className={bubbleClassName}>
                {message.tokens.map((chatToken) => {
                  const tokenIndex = globalTokenIndex;
                  globalTokenIndex++;

                  const {
                    className,
                    styleDict,
                    hasTooltip,
                    tooltipContent,
                    tooltipOpen,
                    tooltipContentClassName,
                    tooltipSide,
                  } = tokenCustomizations[tokenIndex];

                  const tokenElement = (
                    <span
                      key={tokenIndex}
                      className={className}
                      style={styleDict}
                      onMouseDown={(e) => handleMouseDown(tokenIndex, e)}
                      onMouseEnter={() => {
                        handleMouseEnter(tokenIndex);
                        setLocalMousedOverTokenIndex(tokenIndex);
                        dispatch(setMousedOverTokenIndex(tokenIndex));
                      }}
                      onMouseLeave={() => {
                        setLocalMousedOverTokenIndex(null);
                        dispatch(setMousedOverTokenIndex(undefined));
                      }}
                      onMouseUp={handleMouseUp}
                    >
                      {chatToken.token.split('').map((char, i) => {
                        if (char === '\n') {
                          return (
                            <>
                              <span
                                key={i}
                                className="inline-block border border-gray-300 text-xs text-gray-500 font-light px-1"
                              >
                                \n
                              </span>
                              <br />
                            </>
                          );
                        } else {
                          return (
                            <React.Fragment key={i}>{char}</React.Fragment>
                          );
                        }
                      })}
                    </span>
                  );

                  if (hasTooltip) {
                    const openConfig =
                      tooltipOpen !== undefined ? { open: tooltipOpen } : {};
                    return (
                      <Tooltip key={tokenIndex} {...openConfig}>
                        <TooltipTrigger asChild>{tokenElement}</TooltipTrigger>
                        <TooltipContent
                          className={cn(tooltipContentClassName, 'max-w-lg')}
                          side={tooltipSide ?? 'top'}
                        >
                          {tooltipContent}
                        </TooltipContent>
                      </Tooltip>
                    );
                  } else {
                    return tokenElement;
                  }
                })}
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}
