// Skill data configuration file
// Used to manage data for the skill display page

export interface Skill {
	id: string;
	name: string;
	description: string;
	icon: string; // Iconify icon name
	category: "backend" | "tools" | "hardware" | "art" | "sport" | "other";
	level: "beginner" | "intermediate" | "advanced" | "expert";
	experience: {
		years: number;
		months: number;
	};
	projects?: string[]; // Related project IDs
	certifications?: string[];
	color?: string; // Skill card theme color
}

export const skillsData: Skill[] = [
	{
		id: "python",
		name: "Python",
		description:
			"Python编程语言，广泛应用于数据科学、机器学习和Web开发。",
		icon: "logos:python",
		category: "backend",
		level: "intermediate",
		experience: { years: 2, months: 3 },
		projects: [],
		color: "#eeff00ff",
	},
	{
		id: "cpp",
		name: "C++",
		description:
			"C++编程语言，广泛应用于系统编程和高性能计算。",
		icon: "logos:c-plusplus",
		category: "backend",
		level: "intermediate",
		experience: { years: 2, months: 3 },
		projects: [],
		color: "#6093f2ff",
	},
	{
		id: "STM32",
		name: "STM32",
		description:
			"STM32微控制器系列，广泛应用于嵌入式系统和物联网项目。",
		icon: "simple-icons:stmicroelectronics",
		category: "hardware",
		level: "intermediate",
		experience: { years: 0, months: 9 },
		projects: [],
		color: "#1e135aff",
	},
	{
		id: "solidworks",
		name: "SolidWorks",
		description:
			"3D CAD设计软件，广泛应用于机械设计和产品开发领域。",
		icon: "charm:cube",
		category: "tools",
		level: "advanced",
		experience: { years: 1, months: 3 },
		projects: [],
		color: "#590f09ff",
	},
	{
		id: "vscode",
		name: "VS Code",
		description:
			"轻量级但功能强大的源代码编辑器，支持多种编程语言和插件扩展。",
		icon: "logos:visual-studio-code",
		category: "tools",
		level: "intermediate",
		experience: { years: 2, months: 3 },
		projects: [],
		color: "#4acbfaff",
	},
	{
		id: "autocad",
		name: "AutoCAD",
		description:
			"2D/3D CAD设计软件，广泛应用于建筑和工程设计领域。",
		icon: "simple-icons:autocad",
		category: "tools",
		level: "intermediate",
		experience: { years: 0, months: 6 },
		projects: [],
		color: "#b5154aff",
	},
	{
		id: "piano",
		name: "钢琴",
		description:
			"业余爱好。拜厄收尾阶段，车尔尼599三分之一。",
		icon: "material-symbols:piano",
		category: "art",
		level: "beginner",
		experience: { years: 0, months: 8 },
		projects: [],
		color: "#ffffffff",
	},
	{
		id: "paint",
		name: "绘画",
		description:
			"什么时候才有精力重拾呢？",
		icon: "game-icons:paint-brush",
		category: "art",
		level: "beginner",
		experience: { years: 1, months: 0 },
		projects: [],
		color: "#ffdd00ff",
	},
	{
		id: "fitness",
		name: "健身",
		description:
			"(10RM)杠铃卧推35kg，哑铃推肩2*10kg，哑铃划船2*10kg，哑铃弯举2*7.5kg。",
		icon: "material-symbols:fitness-center-rounded",
		category: "sport",
		level: "beginner",
		experience: { years: 1, months: 0 },
		projects: [],
		color: "#6b939fff",
	},
];

// Get skill statistics
export const getSkillStats = () => {
	const total = skillsData.length;
	const byLevel = {
		beginner: skillsData.filter((s) => s.level === "beginner").length,
		intermediate: skillsData.filter((s) => s.level === "intermediate")
			.length,
		advanced: skillsData.filter((s) => s.level === "advanced").length,
		expert: skillsData.filter((s) => s.level === "expert").length,
	};
	const byCategory = {
		backend: skillsData.filter(s => s.category === "backend").length,
		tools: skillsData.filter(s => s.category === "tools").length,
		hardware: skillsData.filter(s => s.category === "hardware").length,
		art: skillsData.filter(s => s.category === "art").length,
		sport: skillsData.filter(s => s.category === "sport").length,
		other: skillsData.filter(s => s.category === "other").length,
	};

	return { total, byLevel, byCategory };
};

// Get skills by category
export const getSkillsByCategory = (category?: string) => {
	if (!category || category === "all") {
		return skillsData;
	}
	return skillsData.filter((s) => s.category === category);
};

// Get advanced skills
export const getAdvancedSkills = () => {
	return skillsData.filter(
		(s) => s.level === "advanced" || s.level === "expert",
	);
};

// Calculate total years of experience
export const getTotalExperience = () => {
	const totalMonths = skillsData.reduce((total, skill) => {
		return total + skill.experience.years * 12 + skill.experience.months;
	}, 0);
	return {
		years: Math.floor(totalMonths / 12),
		months: totalMonths % 12,
	};
};
