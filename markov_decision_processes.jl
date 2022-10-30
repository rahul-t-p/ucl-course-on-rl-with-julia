### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 32942b67-a606-4fce-8795-c333b2120561
using PlutoUI

# ╔═╡ f4a6f2cb-fea6-48d1-8d79-6d04136f25a9
using LinearAlgebra

# ╔═╡ fe0799a0-50fa-11ed-0216-91b8438180f7
md"""
# Markov Decision Processes
"""

# ╔═╡ d651a091-e0cb-440d-82a5-2f03259525fc
md"""
## Student Markov Chain
"""

# ╔═╡ 51fe67ba-7b8e-4c70-9ae4-5559a1ed0fef
md"""
### Environment
Resources from [UCL Course on RL](https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf)
"""

# ╔═╡ da82a42d-56dc-4795-82b8-61b79189ab8b
PlutoUI.LocalResource("src/l2_1.png")

# ╔═╡ 94720b05-98c9-449c-9949-b140919517c6
PlutoUI.LocalResource("src/l2_2.png")

# ╔═╡ fbdfe150-cc97-42e9-b160-f5de8fccfa9e
ENV = Dict(
	"Facebook" => Dict(
		"Facebook" => [
			(0.9, "Facebook", -1, false),
			(0.1, "Class1", 0, false),
		],
		"Quit" => [
			(0.9, "Facebook", -1, false),
			(0.1, "Class1", 0, false),
		],
	),
	"Class1" => Dict(
		"Facebook" => [
			(0.5, "Facebook", -1, false),
			(0.5, "Class2", -2, false),
		],
		"Study" => [
			(0.5, "Facebook", -1, false),
			(0.5, "Class2", -2, false),
		],
	),
	"Class2" => Dict(
		"Sleep" => [
			(0.2, "Sleep", 0, true),
			(0.8, "Class3", -2, false),
		],
		"Study" => [
			(0.2, "Sleep", 0, true),
			(0.8, "Class3", -2, false),
		],
	),
	"Class3" => Dict(
		"Study" => [
			(0.6, "Sleep", 10, true),
			(0.4*0.4, "Class3", 1, false),
			(0.4*0.4, "Class2", 1, false),
			(0.4*0.2, "Class1", 1, false),
		],
		"Pub" => [
			(0.6, "Sleep", 10, true),
			(0.4*0.4, "Class3", 1, false),
			(0.4*0.4, "Class2", 1, false),
			(0.4*0.2, "Class1", 1, false),
		],
	),
	"Sleep" => Dict(
		"Sleep" => [
			(1.0, "Sleep", 0, true),
		],
	),
)

# ╔═╡ 693b875d-fa1d-4fc6-93f3-ec145b796c1a
md"""
### Direct Solution
```math
v = R + \gamma Pv
```
```math
(I - \gamma P)v = R
```
```math
v = (I - \gamma P)^{-1} R
```
"""

# ╔═╡ cf8c8f01-c68c-41bc-bd8b-ed660faeb458
#=
Define γ with default value 0.9.
Note: When γ is 1 inverse of (I - γ*P) does not exist.
=#
@bind γ Slider(0:0.1:1, default=0.9, show_value=true)

# ╔═╡ 201d4f23-bf64-480c-bc7c-fc12e2b57344
#=
Define Transition probability matrix;
		facebook class1 class2 class3 sleep
facebook  ...     ...    ...    ...    ...
class1    ...     ...    ...    ...    ...
class2    ...     ...    ...    ...    ...
class3    ...     ...    ...    ...    ...
sleep     ...     ...    ...    ...    ...
=#

P = [[0.9 0.1 0 0 0]
	 [0.5 0 0.5 0 0]
	 [0 0 0 0.8 0.2]
	 [0 0.08 0.16 0.16 0.6]
	 [0 0 0 0 1]]

# ╔═╡ 98c75c4b-c022-4acf-acfe-127215b24967
#=
Define Reward matrix
Reward is column vector with one entry per state.
Refer Slide 22 of https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf
=#
R = [-1
	 -2
	 -2
	 -2
	 10]

# ╔═╡ 3ee62af7-01f5-479a-a657-341c5f93843d
v = inv(I - γ*P) * R

# ╔═╡ c0173e9a-b66e-4a1e-b0b3-d24eba2c8abb
md"""
### State-Value function
"""

# ╔═╡ 8c671122-4a6d-40e9-8cad-399f54ea0181
md"#### for π(a/s) = 0.5"

# ╔═╡ 8a64c5ea-07ae-4920-b5be-1fa3b83a2552
begin
	state_val_function = Dict([state => 0. for state in keys(ENV)])
	for _ in 1:100
		for state in keys(ENV)
			state_val = 0
			for action in keys(ENV[state])
				for transition in ENV[state][action]
					p, next_state, r, done = transition
					state_val += 0.5*(r+γ*p*(state_val_function[next_state]))
				end
			end
			state_val_function[state] = state_val
		end
	end
end

# ╔═╡ c193814a-f587-4f9a-b701-da806bb30be3
state_val_function

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.48"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "6c01a9b494f6d2a9fc180a08b182fcb06f0958a0"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "efc140104e6d0ae3e7e30d56c98c4a927154d684"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.48"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─fe0799a0-50fa-11ed-0216-91b8438180f7
# ╟─d651a091-e0cb-440d-82a5-2f03259525fc
# ╟─51fe67ba-7b8e-4c70-9ae4-5559a1ed0fef
# ╟─32942b67-a606-4fce-8795-c333b2120561
# ╟─da82a42d-56dc-4795-82b8-61b79189ab8b
# ╟─94720b05-98c9-449c-9949-b140919517c6
# ╟─fbdfe150-cc97-42e9-b160-f5de8fccfa9e
# ╟─693b875d-fa1d-4fc6-93f3-ec145b796c1a
# ╠═f4a6f2cb-fea6-48d1-8d79-6d04136f25a9
# ╠═cf8c8f01-c68c-41bc-bd8b-ed660faeb458
# ╠═201d4f23-bf64-480c-bc7c-fc12e2b57344
# ╠═98c75c4b-c022-4acf-acfe-127215b24967
# ╠═3ee62af7-01f5-479a-a657-341c5f93843d
# ╟─c0173e9a-b66e-4a1e-b0b3-d24eba2c8abb
# ╟─8c671122-4a6d-40e9-8cad-399f54ea0181
# ╠═8a64c5ea-07ae-4920-b5be-1fa3b83a2552
# ╠═c193814a-f587-4f9a-b701-da806bb30be3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
