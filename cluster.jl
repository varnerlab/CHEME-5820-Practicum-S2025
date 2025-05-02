using CSV
using DataFrames
using TSne
using Clustering
using Plots

# Load the embeddings CSV
df = CSV.read("data/sample_english_word_embeddings.csv", DataFrame)

# Prepare data matrix (remove word column)
X = Matrix(df[:, Not(:word)])

# Run t-SNE (100D -> 2D)
# Old API: tsne(data, no_dims, initial_dims, perplexity, theta)
# We'll set: no_dims=2, initial_dims=100 (full), perplexity=30, theta=0.5
Y = tsne(X, 2, size(X, 2))

# # Run k-means clustering
# k = 10
# kmeans_result = kmeans(X', k; maxiter=100, display=:none)

# # Extract cluster labels
# cluster_assignments = kmeans_result.assignments

# # Find the closest word to each cluster center
# function closest_word(center, data)
#     dists = sum((data .- center').^2, dims=2)
#     return argmin(dists)
# end

# centroids = kmeans_result.centers
# closest_indices = [closest_word(centroids[i, :], X) for i in 1:k]

# # Prepare the plot
# scatter(
#     Y[:, 1], Y[:, 2],
#     group=cluster_assignments,
#     title="t-SNE of 10,000 Word Embeddings (K-Means Clusters with Labels)",
#     xlabel="t-SNE Dimension 1",
#     ylabel="t-SNE Dimension 2",
#     markersize=2,
#     legend=false,
# )

# # Add labels for closest words
# for idx in closest_indices
#     annotate!(
#         Y[idx, 1], Y[idx, 2],
#         text(string(df.word[idx]), :black, :bold, 8),
#     )
# end

# # Save to PDF
# savefig("tsne_clusters_with_labels.pdf")
