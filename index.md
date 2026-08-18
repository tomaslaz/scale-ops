---
layout: distill
title: "Scale Ops"
subtitle: "Exploring how AI models scale"
description: "ScaleOps is a repository dedicated to collecting and sharing findings from exploring and optimizing the scalability and performance of AI workloads."
future: true
htmlwidgets: true
hidden: false

giscus_comments: false

# section_number: 0

# previous_section_url: ""
# previous_section_name: "Introduction"

# next_section_url: net_1
# next_section_name: "Networking Part 1"

# bibliography: main.bib

# citation: true

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Goals
  - name: Links to Sections

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Goals

I warmly welcome any feedback or discussion, especially if you spot potential oversights in my reasoning or experiments.

I hope that this repository will be a useful resource for anyone who is learning about or interested in the performance and scalability of AI workloads.

## Links to Sections

{% assign grouped_posts = site.posts | group_by: "category" %}
{% for group in grouped_posts %}
**{{ group.name }}**

{% assign sorted_posts = group.items | sort: "date" | reverse %}
{% for post in sorted_posts %}

- [**{{ post.title }}**]({{ post.url | relative_url }}) _({{ post.date | date: "%Y-%m-%d" }})_ {{ post.description }}

{% endfor %}
{% endfor %}
