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
            },{id: "dropdown-networking-part-1",
              title: "Networking Part 1",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/scale-ops/net_1";
              },
            },{id: "dropdown-networking-part-2",
              title: "Networking Part 2",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/scale-ops/net_2";
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
