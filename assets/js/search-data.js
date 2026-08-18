// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-",
    title: "",
    section: "Navigation",
    handler: () => {
      window.location.href = "/scale-ops/";
    },
  },{id: "dropdown-introduction",
              title: "Introduction",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/scale-ops/index";
              },
            },{id: "post-performance-characterisation-of-anemoi-training-on-isambard-ai",
      
        title: "Performance Characterisation of Anemoi Training on Isambard-AI",
      
      description: "A systematic performance investigation of Anemoi weather model training on Isambard-AI GH200 nodes, from a single GPU up to 100 nodes (400 GPUs), covering roofline analysis, NCCL benchmarking, and multi-node scaling efficiency.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/scale-ops/anemoi_isambard/";
        
      },
    },{id: "post-the-impact-of-gpu-networking-part-2",
      
        title: "The Impact of GPU Networking (Part 2)",
      
      description: "Part 2 builds on earlier experiments by examining how distributing 4 GPUs across 1, 2, and 4 nodes impacts transformer model training, with a focus on network topology and NIC sharing.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/scale-ops/net_2/";
        
      },
    },{id: "post-the-impact-of-gpu-networking-part-1",
      
        title: "The Impact of GPU Networking (Part 1)",
      
      description: "This post examines the impact of GPU networking on transformer model training performance using Distributed Data Parallel (DDP), comparing high-speed intra-node NVLink with slower inter-node InfiniBand.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/scale-ops/net_1/";
        
      },
    },{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/scale-ops/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
