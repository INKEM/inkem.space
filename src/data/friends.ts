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
	{
		id: 3,
		title: "人言兑",
		imgurl: "https://pic.rmb.bdstatic.com/bjh/3eacca401b/250205/c9fae3c96b0da42d4d5a2a5ebcf8a1c8.png?x-bce-process=image/resize,m_lfit,w_1242",
		desc: "人言成信 言兑才说",
		siteurl: "https://blog.axiaoxin.com/",
		tags: ["我关注的"],
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
