import { createRouter, createWebHistory } from 'vue-router'
import Phonebook from '../components/Phonebook.vue'

const routes = [
  {
    path: '/',
    name: 'Phonebook',
    component: Phonebook
  }
]

const router = createRouter({
  // 使用静态字符串替换 process.env.BASE_URL
  history: createWebHistory('/'),
  routes
})

export default router
