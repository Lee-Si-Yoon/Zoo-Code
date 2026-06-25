import { type FriendliModelId, friendliDefaultModelId, friendliModels } from "@roo-code/types"

import type { ApiHandlerOptions } from "../../shared/api"

import { BaseOpenAiCompatibleProvider } from "./base-openai-compatible-provider"

export class FriendliHandler extends BaseOpenAiCompatibleProvider<FriendliModelId> {
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
}
