import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI from "openai"

import { type FriendliModelId, friendliDefaultModelId, friendliModels } from "@roo-code/types"

import type { ApiHandlerOptions } from "../../shared/api"
import { shouldUseReasoningEffort, getModelMaxOutputTokens } from "../../shared/api"

import { convertToOpenAiMessages } from "../transform/openai-format"
import { getModelParams } from "../transform/model-params"

import { BaseOpenAiCompatibleProvider } from "./base-openai-compatible-provider"
import { handleOpenAIError } from "./utils/error-handler"
import type { ApiHandlerCreateMessageMetadata } from "../index"

/**
 * Friendli extends the OpenAI Chat Completions API with these non-standard fields:
 * - reasoning_effort: enum (minimal, low, medium, high, xhigh, max) — reasoning depth
 * - chat_template_kwargs: { enable_thinking: boolean } — toggles thinking for controllable models
 * - parse_reasoning / include_reasoning: when true, Friendli streams reasoning via
 *   delta.reasoning_content (which extractReasoningFromDelta already handles)
 * - reasoning_budget: integer token budget (not currently surfaced in settings UI)
 */
type FriendliChatCompletionParams = OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming & {
	chat_template_kwargs?: { enable_thinking: boolean }
	parse_reasoning?: boolean
	include_reasoning?: boolean
	// Friendli's reasoning_effort supports a broader enum than OpenAI's type allows
	reasoning_effort?:
		| OpenAI.Chat.Completions.ChatCompletionCreateParams["reasoning_effort"]
		| "minimal"
		| "xhigh"
		| "max"
}

/**
 * Handler for the Friendli Model APIs (OpenAI-compatible).
 * Routes chat completions to `https://api.friendli.ai/serverless/v1`.
 *
 * Overrides `createStream` and `completePrompt` to inject Friendli-specific
 * reasoning parameters that the base class doesn't know about.
 */
export class FriendliHandler extends BaseOpenAiCompatibleProvider<FriendliModelId> {
	/**
	 * @param options  Provider settings; `friendliApiKey` is required.
	 */
	constructor(options: ApiHandlerOptions) {
		super({
			...options,
			providerName: "Friendli",
			baseURL: "https://api.friendli.ai/serverless/v1",
			apiKey: options.friendliApiKey,
			defaultProviderModelId: friendliDefaultModelId,
			providerModels: friendliModels,
			defaultTemperature: 0.6,
		})
	}

	override getModel() {
		const id =
			this.options.apiModelId && this.options.apiModelId in this.providerModels
				? (this.options.apiModelId as FriendliModelId)
				: this.defaultProviderModelId

		const info = this.providerModels[id]
		const params = getModelParams({
			format: "openai",
			modelId: id,
			model: info,
			settings: this.options,
			defaultTemperature: 0.6,
		})

		return { id, info, ...params }
	}

	/**
	 * Build Friendli-specific reasoning params to merge into the OpenAI request.
	 *
	 * Rules:
	 * - Controllable reasoning models (GLM-5.x): send `reasoning_effort` + `chat_template_kwargs.enable_thinking`
	 *   when user selected a valid effort. When user disabled reasoning (none/disable), send nothing —
	 *   the model runs at its default behavior.
	 * - Always-reasoning models (MiniMax-M2.5): send `parse_reasoning` + `include_reasoning` only
	 * - Non-reasoning models (DeepSeek-V3.2): no extra params (reasoning_effort silently ignored)
	 */
	private buildFriendliReasoningParams(): Partial<FriendliChatCompletionParams> {
		const { info: modelInfo, reasoningEffort } = this.getModel()
		const extra: Partial<FriendliChatCompletionParams> = {}

		const useReasoningEffort = modelInfo.supportsReasoningEffort
			? shouldUseReasoningEffort({ model: modelInfo, settings: this.options })
			: false

		const useReasoningBinary =
			!!(modelInfo.supportsReasoningBinary && this.options.enableReasoningEffort)

		// No reasoning support or user explicitly disabled — send nothing, let model use its default
		if (!useReasoningEffort && !useReasoningBinary) {
			return extra
		}

		// parse_reasoning + include_reasoning: enable for all reasoning-enabled cases
		extra.parse_reasoning = true
		extra.include_reasoning = true

		// reasoning_effort: for controllable reasoning models (GLM-5.x)
		// getModelParams already filters out "disable" and "none", so reasoningEffort
		// from getModel() is a valid effort value or undefined.
		if (useReasoningEffort && reasoningEffort) {
			extra.reasoning_effort = reasoningEffort as FriendliChatCompletionParams["reasoning_effort"]
		}

		// chat_template_kwargs: only for controllable reasoning models (GLM-5.x)
		// MiniMax-M2.5 is always-reasoning — no enable_thinking toggle
		if (useReasoningEffort && Array.isArray(modelInfo.supportsReasoningEffort)) {
			extra.chat_template_kwargs = { enable_thinking: true }
		}

		return extra
	}

	/**
	 * Override createStream to inject Friendli-specific reasoning params.
	 * The base class createMessage() calls createStream and handles all stream
	 * processing (TagMatcher, extractReasoningFromDelta, tool calls, usage).
	 */
	protected override createStream(
		systemPrompt: string,
		messages: Anthropic.Messages.MessageParam[],
		metadata?: ApiHandlerCreateMessageMetadata,
		requestOptions?: OpenAI.RequestOptions,
	) {
		const friendliExtra = this.buildFriendliReasoningParams()

		const { id: model, info } = this.getModel()

		// Centralized cap: clamp to 20% of the context window
		const max_tokens = getModelMaxOutputTokens({
			modelId: model,
			model: info,
			settings: this.options,
			format: "openai",
		}) ?? undefined

		const temperature = this.options.modelTemperature ?? info.defaultTemperature ?? this.defaultTemperature

		const params: FriendliChatCompletionParams = {
			model,
			max_tokens,
			temperature,
			messages: [{ role: "system", content: systemPrompt }, ...convertToOpenAiMessages(messages)],
			stream: true,
			stream_options: { include_usage: true },
			tools: this.convertToolsForOpenAI(metadata?.tools),
			tool_choice: metadata?.tool_choice,
			parallel_tool_calls: metadata?.parallelToolCalls ?? true,
			...friendliExtra,
		}

		try {
			return this.client.chat.completions.create(
				params as OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming,
				requestOptions,
			)
		} catch (error) {
			throw handleOpenAIError(error, this.providerName)
		}
	}

	override async completePrompt(prompt: string): Promise<string> {
		const { id: modelId, info: modelInfo } = this.getModel()
		const friendliExtra = this.buildFriendliReasoningParams()

		const params: OpenAI.Chat.Completions.ChatCompletionCreateParams & Partial<FriendliChatCompletionParams> = {
			model: modelId,
			messages: [{ role: "user", content: prompt }],
			...friendliExtra,
		}

		try {
			const response = await this.client.chat.completions.create(params as any)

			// Check for provider-specific error responses (e.g., MiniMax base_resp)
			const responseAny = response as any
			if (responseAny.base_resp?.status_code && responseAny.base_resp.status_code !== 0) {
				throw new Error(
					`${this.providerName} API Error (${responseAny.base_resp.status_code}): ${responseAny.base_resp.status_msg || "Unknown error"}`,
				)
			}

			return response.choices?.[0]?.message.content || ""
		} catch (error) {
			throw handleOpenAIError(error, this.providerName)
		}
	}
}
