### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ ee3014ce-5c24-11ed-2358-d550572c89a8
md"""
# Model-Free Control
"""

# ╔═╡ d670cf69-f4fd-4008-a2f6-79bdffc01bea
md"""
Resources from [UCL Course on RL](https://www.davidsilver.uk/wp-content/uploads/2020/03/control.pdf)
"""

# ╔═╡ 31987c6f-9aaf-44da-9bf3-b7adb5112236
md"""
## Blackjack
"""

# ╔═╡ cb5aa044-a714-4556-9da1-ead74c7638d1
md"""
### Model-Free Prediction
"""

# ╔═╡ 72ae8f38-f581-4326-b709-c89555d03078
begin # Blackjack env

	@enum ACTION begin
			HIT = 1
			STICK
	end
	
	Base.@kwdef mutable struct BLACKJACK
		A = 1
		REWARDS = Dict("WIN" => 1, "LOSE" => -1, "DRAW" => 0)
		ACTIONS = ACTION
		DECK = Dict{String, Int}(
				"A" => A,
				"1" => 1,
				"2" => 2,
				"3" => 3,
				"4" => 4,
				"5" => 5,
				"6" => 6,
				"7" => 7,
				"8" => 8,
				"9" => 9,
				"10" => 10,
				"Jack" => 10,
				"Queen" => 10,
				"King" => 10,
			)
		OBSERVATION = Dict("Player" => Vector{Int}(), "Dealer" => Vector{Int}())
		HIDDEN_CARD = nothing
	end
	
	function create_blackjack_env(; usable=false)
		return (usable ? BLACKJACK(A=11) : BLACKJACK())
	end

	function reset!(env::BLACKJACK)
		env.OBSERVATION = Dict("Player" => Vector{Int}(), "Dealer" => Vector{Int}())
		for _ in 1:2
			append!(env.OBSERVATION["Player"], env.DECK[rand(keys(env.DECK))])
			append!(env.OBSERVATION["Dealer"], env.DECK[rand(keys(env.DECK))])
		end
		env.HIDDEN_CARD = env.OBSERVATION["Dealer"][end]
		env.OBSERVATION["Dealer"] = env.OBSERVATION["Dealer"][1:end-1]
		return env.OBSERVATION
	end

	function step!(env::BLACKJACK, action)
		if sum(env.OBSERVATION["Player"]) == 21 # Blackjack
			return env.OBSERVATION, env.REWARDS["WIN"], true
		end
		
		if action == HIT
			append!(env.OBSERVATION["Player"], env.DECK[rand(keys(env.DECK))])
			if sum(env.OBSERVATION["Player"]) > 21 # Player Busted
				return env.OBSERVATION, env.REWARDS["LOSE"], true
			end
		elseif action == STICK
			append!(env.OBSERVATION["Dealer"], env.HIDDEN_CARD)
			while sum(env.OBSERVATION["Dealer"]) < 17 # Dealer's strategy
				append!(env.OBSERVATION["Dealer"], env.DECK[rand(keys(env.DECK))])
			end
			if sum(env.OBSERVATION["Dealer"]) > 21 # Dealer Busted
				return env.OBSERVATION, env.REWARDS["WIN"], true
			end
		end
		
		if sum(env.OBSERVATION["Player"]) > sum(env.OBSERVATION["Dealer"]) # Player Wins
			return env.OBSERVATION, env.REWARDS["WIN"], true
		elseif sum(env.OBSERVATION["Player"]) < sum(env.OBSERVATION["Dealer"]) # Dealer Wins
			return env.OBSERVATION, env.REWARDS["LOSE"], true
		elseif sum(env.OBSERVATION["Player"]) == sum(env.OBSERVATION["Dealer"]) # Game Draw
			return env.OBSERVATION, env.REWARDS["DRAW"], true
		end
		
		return env.OBSERVATION, 0, false
	end
end

# ╔═╡ abadbbc4-c170-4812-b414-56a7daed271e
begin
	BJ_USABLE = create_blackjack_env(usable=true)
	for i in 1:10
		println("Episode $i")
		obs = reset!(BJ_USABLE)
		println(obs)
		done = false
		while !done
			action = rand(instances(BJ_USABLE.ACTIONS))
			println(action)
			next_state, r, done = step!(BJ_USABLE, action)
			println("$next_state, $r, $done")
		end
	end
end

# ╔═╡ Cell order:
# ╟─ee3014ce-5c24-11ed-2358-d550572c89a8
# ╟─d670cf69-f4fd-4008-a2f6-79bdffc01bea
# ╟─31987c6f-9aaf-44da-9bf3-b7adb5112236
# ╟─cb5aa044-a714-4556-9da1-ead74c7638d1
# ╠═72ae8f38-f581-4326-b709-c89555d03078
# ╠═abadbbc4-c170-4812-b414-56a7daed271e
