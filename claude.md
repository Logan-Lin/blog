I am in the process of migrating content from my previous quartz 4-based blog site in /Users/yanlin/Documents/Projects/personal-blog to this Zola-based blog post.

For each blog post, follow the following process of migration:
1. Create an empty bundle (directory with the same name as the old markdown file) and under section of the same original one
2. Copy the old markdown file to the bundle as index.md (first copy the file using `cp` directly, then edit)
3. Edit the frontmatter:

```
+++
title = "(the original title)"
date = (the old created field)
description = "(leave blank)"
+++
```

4. Find, copy, and rename the images used in the post to the bundle
5. Replace the old Obsidian-flavor markdown links (images ![[]] and internal links [[]]) with standard markdown links
6. Turn callout blocks into standard markdown quote blocks, e.g., >[!note], >[!TLDR], >[!quote] → > **Note:**, > **TL;DR:**, > **References:**; e.g. > [!tip] Videos -> > **Videos:**, > [!info] Extended Reading -> > **Extended Reading**
7. For multiline math equations (those with \\), wrap the whole equation like below to avoid Zola's processing:

```
{% math() %}
f_{\{q,k\}}(x_m, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} W_{\{q,k\}}^{(11)} & W_{\{q,k\}}^{(12)} \\ W_{\{q,k\}}^{(21)} & W_{\{q,k\}}^{(22)} \end{pmatrix} \begin{pmatrix} x_m^{(1)} \\ x_m^{(2)} \end{pmatrix}
{% end %}
```

