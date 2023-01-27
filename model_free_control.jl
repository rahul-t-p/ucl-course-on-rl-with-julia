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
		OBSERVATION = Dict("Player" => Vector{Int}(), "Dealer" => Vector{String}())
		HIDDEN_CARD = nothing
	end
	
	function create_blackjack_env(; usable=false)
		return (usable ? BLACKJACK(A=11) : BLACKJACK())
	end

	function reset!(env::BLACKJACK)
		done = false
		env.OBSERVATION = Dict("Player" => Vector{Int}(), "Dealer" => Vector{String}())
		for _ in 1:2
			append!(env.OBSERVATION["Player"], env.DECK[rand(keys(env.DECK))])
			append!(env.OBSERVATION["Dealer"], [rand(keys(env.DECK))])
		end
		env.HIDDEN_CARD = env.OBSERVATION["Dealer"][end]
		env.OBSERVATION["Dealer"] = env.OBSERVATION["Dealer"][1:end-1]
		if sum(env.OBSERVATION["Player"]) >= 21 # Game over
			done = true
		end
		return (sum(env.OBSERVATION["Player"]), env.OBSERVATION["Dealer"][1]), done
	end

	function step!(env::BLACKJACK, action)
		if sum(env.OBSERVATION["Player"]) == 21 # Blackjack
			return (sum(env.OBSERVATION["Player"]), env.OBSERVATION["Dealer"][1]), env.REWARDS["WIN"], true
		end
		
		if action == HIT
			append!(env.OBSERVATION["Player"], env.DECK[rand(keys(env.DECK))])
			if sum(env.OBSERVATION["Player"]) > 21 # Player Busted
				return (sum(env.OBSERVATION["Player"]), env.OBSERVATION["Dealer"][1]), env.REWARDS["LOSE"], true
			end
		elseif action == STICK
			append!(env.OBSERVATION["Dealer"], [env.HIDDEN_CARD])
			while sum([env.DECK[i] for i in env.OBSERVATION["Dealer"]]) < 17 # Dealer's strategy
				append!(env.OBSERVATION["Dealer"], [rand(keys(env.DECK))])
			end
			if sum([env.DECK[i] for i in env.OBSERVATION["Dealer"]]) > 21 # Dealer Busted
				return (sum(env.OBSERVATION["Player"]), env.OBSERVATION["Dealer"][1]), env.REWARDS["WIN"], true
			end
			if sum(env.OBSERVATION["Player"]) > sum([env.DECK[i] for i in env.OBSERVATION["Dealer"]]) # Player Wins
				return (sum(env.OBSERVATION["Player"]), env.OBSERVATION["Dealer"][1]), env.REWARDS["WIN"], true
			elseif sum(env.OBSERVATION["Player"]) < sum([env.DECK[i] for i in env.OBSERVATION["Dealer"]]) # Dealer Wins
				return (sum(env.OBSERVATION["Player"]), env.OBSERVATION["Dealer"][1]), env.REWARDS["LOSE"], true
			elseif sum(env.OBSERVATION["Player"]) == sum([env.DECK[i] for i in env.OBSERVATION["Dealer"]]) # Game Draw
				return (sum(env.OBSERVATION["Player"]), env.OBSERVATION["Dealer"][1]), env.REWARDS["DRAW"], true
			end
		end
		
		return (sum(env.OBSERVATION["Player"]), env.OBSERVATION["Dealer"][1]), 0, false
	end
end

# ╔═╡ abadbbc4-c170-4812-b414-56a7daed271e
begin
	# Create usable ace env
	BJ_USABLE = create_blackjack_env(usable=true)
	# Run few random episodes to check the implementation
	for i in 1:5
		println("Episode $i")
		obs, done = reset!(BJ_USABLE)
		println(obs)
		while !done
			action = rand(instances(BJ_USABLE.ACTIONS))
			println(action)
			next_state, r, done = step!(BJ_USABLE, action)
			println("$next_state, $r, $done")
		end
	end
end

# ╔═╡ f0d71d85-5f1c-45cd-b40b-003e73f21647
begin
	# Create no usable ace env
	BJ = create_blackjack_env()
	# Run few random episodes to check the implementation
	for i in 1:5
		println("Episode $i")
		obs, done = reset!(BJ_USABLE)
		println(obs)
		while !done
			action = rand(instances(BJ_USABLE.ACTIONS))
			println(action)
			next_state, r, done = step!(BJ_USABLE, action)
			println("$next_state, $r, $done")
		end
	end
end

# ╔═╡ cb5aa044-a714-4556-9da1-ead74c7638d1
md"""
### GLIE Monte-Carlo Control
"""

# ╔═╡ 023a64f4-8e47-4d26-9d0c-2feffc75f713
# GLIE Monte-Carlo Control
function GLIE_MC_Control(env)
	γ = 1
	Q = Dict(
		((i, j), act) => 0.
		for i in 2:21 for j in keys(env.DECK) for act in instances(env.ACTIONS)
	)
	# Initializing an arbitrary policy to HIT all the time
	π = Dict(
		(i, j) => HIT
		for i in 2:21 for j in keys(env.DECK)
	)
	ϵ = 1.
	for episode in 1:5_00_000
		state, done = reset!(env) # Starting state
		if done
			continue
		end
		# ϵ = max(1/episode, 0.01)
		if episode%5_000 == 0
			ϵ = ϵ-0.01
		end
		trajectory = Vector{}()
		N = Dict(
			((i, j), act) => 0
			for i in 2:21 for j in keys(env.DECK) for act in instances(env.ACTIONS)
		)
		while !done # Update trajectory
			# ϵ-greedy policy
			# action = π[state]
			if ϵ > rand()
				action = rand(instances(env.ACTIONS))
			else
				q_val = -Inf
				for act in instances(env.ACTIONS)
					if Q[(state, act)] > q_val
						q_val = Q[(state, act)]
						action = act
					end
				end
			end
			next_state, r, done = step!(env, action)
			append!(trajectory, [(state, action, r)])
			state = next_state
			if done
				break
			end
		end
		# Update Q table
		for (t, (state, action, r)) in enumerate(trajectory)
			# Increment N
			N[(state, action)] += 1
			# Calculate return starting from that state
			discounts = [γ^i for i in 0:length(trajectory)-t]
			rewards = map(x->x[3], trajectory[t:end])
			Gₜ = sum(discounts .* rewards)
			# Update action value fuction incrementaly
			Q[(state, action)] += (1/N[(state, action)]) * (Gₜ - Q[(state, action)])
		end
		# Improve Policy
		for state in keys(copy(π))
			if ϵ > rand()
				action = rand(instances(env.ACTIONS))
			else
				q_val = -Inf
				for act in instances(env.ACTIONS)
					if Q[(state, act)] > q_val
						q_val = Q[(state, act)]
						action = act
					end
				end
			end
			π[state] = action
		end
	end
	return Q, π
end

# ╔═╡ 4b38afe9-3d51-40a5-abf4-15d138eb7949
Q, π = GLIE_MC_Control(BJ)

# ╔═╡ 7ab1e899-3069-41c3-b230-0f40f1143edd
Q[((20, "A"), STICK)], Q[((20, "A"), HIT)]

# ╔═╡ 8dbc3a0b-b218-4435-ba48-b58b2a1d5c5f
Q_USABLE, π_USABLE = GLIE_MC_Control(BJ_USABLE)

# ╔═╡ 8f212539-2ad2-42a5-af06-ce7961077262
for (state, action) in π_USABLE
	if action == STICK
		println(state)
	end
end

# ╔═╡ fcdacf9a-a6a8-428a-8643-912c54522c84
Q_USABLE[((20, "9"), STICK)], Q_USABLE[((20, "9"), HIT)]

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[deps]
"""

# ╔═╡ Cell order:
# ╟─ee3014ce-5c24-11ed-2358-d550572c89a8
# ╟─d670cf69-f4fd-4008-a2f6-79bdffc01bea
# ╟─31987c6f-9aaf-44da-9bf3-b7adb5112236
# ╠═72ae8f38-f581-4326-b709-c89555d03078
# ╠═abadbbc4-c170-4812-b414-56a7daed271e
# ╠═f0d71d85-5f1c-45cd-b40b-003e73f21647
# ╟─cb5aa044-a714-4556-9da1-ead74c7638d1
# ╠═023a64f4-8e47-4d26-9d0c-2feffc75f713
# ╠═4b38afe9-3d51-40a5-abf4-15d138eb7949
# ╠═7ab1e899-3069-41c3-b230-0f40f1143edd
# ╠═8dbc3a0b-b218-4435-ba48-b58b2a1d5c5f
# ╠═8f212539-2ad2-42a5-af06-ce7961077262
# ╠═fcdacf9a-a6a8-428a-8643-912c54522c84
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
