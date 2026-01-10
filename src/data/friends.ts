// 友情链接数据配置
// 用于管理友情链接页面的数据

export interface FriendItem {
	id: number;
	title: string;
	imgurl: string;
	desc: string;
	siteurl: string;
	tags: string[];
}

// 友情链接数据
export const friendsData: FriendItem[] = [
	{
		id: 1,
		title: "时歌的博客",
		imgurl: "https://www.lapis.cafe/avatar.webp",
		desc: "理解以真实为本，但真实本身并不会自动呈现",
		siteurl: "https://www.lapis.cafe",
		tags: ["我关注的"],
	},
	{
		id: 2,
		title: "Eagle's Blog",
		imgurl: "https://patchouli.group/_astro/avatar.CJBAIyXG_cmwQ1.webp",
		desc: "身无彩凤双飞翼，心有灵犀一点通。",
		siteurl: "https://patchouli.group/",
		tags: ["我的朋友"],
	},
];

// 获取所有友情链接数据
export function getFriendsList(): FriendItem[] {
	return friendsData;
}

// 获取随机排序的友情链接数据
export function getShuffledFriendsList(): FriendItem[] {
	const shuffled = [...friendsData];
	for (let i = shuffled.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
	}
	return shuffled;
}
