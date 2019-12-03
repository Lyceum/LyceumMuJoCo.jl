using Documenter, LyceumMuJoCo

const PAGES = [
    "Home" => "index.md",
    "API" => "api.md"
]

makedocs(;
    modules = [LyceumMuJoCo],
    format = Documenter.HTML(prettyurls = get(ENV, "GITHUB_ACTIONS", nothing) == "true"),
    pages = PAGES,
    sitename = "LyceumMuJoCo.jl",
    authors = "Colin Summers",
    clean = true,
    doctest = true,
    checkdocs = :exports,
    linkcheck = :true,
    linkcheck_ignore = [r"^https://github.com/Lyceum/.*/actions"],
)

deploydocs(repo = "github.com/Lyceum/LyceumMuJoCo.jl.git")
